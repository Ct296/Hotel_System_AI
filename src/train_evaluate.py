import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config
from config import (
    TOKENIZED_DATA_FILE, BEST_MODEL_FILE, METRICS_DIR,
    MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, LABEL_CLASSES_FILE
)

# Lấy các tham số nâng cao (tự động nhận giá trị mặc định nếu config.py chưa khai báo)
WEIGHT_DECAY = getattr(config, "WEIGHT_DECAY", 0.01)
EARLY_STOPPING_PATIENCE = getattr(config, "EARLY_STOPPING_PATIENCE", 3)
EARLY_STOPPING_DELTA = getattr(config, "EARLY_STOPPING_DELTA", 0.001)
ASPECT_LOSS_WEIGHTS = getattr(config, "ASPECT_LOSS_WEIGHTS", [0.30, 0.90, 1.40, 1.10])
CALIBRATION_WEIGHTS = getattr(config, "CALIBRATION_WEIGHTS", [1.00, 0.95, 1.30, 1.10])


# 1. KIẾN TRÚC MÔ HÌNH MULTI-TASK PHOBERT
class HotelCommentModel(nn.Module):
    def __init__(self, num_sentiment_classes, num_aspects, num_polarities=4):
        super(HotelCommentModel, self).__init__()
        self.num_aspects = num_aspects
        self.num_polarities = num_polarities
        
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.phobert.config.hidden_size
        
        # Kết hợp CLS token và Mean Pooling (768 * 2 = 1536)
        combined_size = hidden_size * 2
        self.drop = nn.Dropout(p=0.3)
        
        # Nhánh 1: Dự đoán sắc thái tổng thể (3 lớp)
        self.sentiment_head = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_sentiment_classes)
        )
        
        # Nhánh 2: Dự đoán đồng thời N khía cạnh x 4 trạng thái (none, pos, neg, neu)
        self.aspect_head = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_aspects * num_polarities)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state
        
        cls_output = seq_output[:, 0, :]
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(seq_output.size()).float()
        sum_embeddings = torch.sum(seq_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_output = sum_embeddings / sum_mask
        
        pooled_output = self.drop(torch.cat([cls_output, mean_output], dim=1))
        
        sentiment_logits = self.sentiment_head(pooled_output)
        aspect_logits = self.aspect_head(pooled_output)
        aspect_logits = aspect_logits.view(-1, self.num_aspects, self.num_polarities)
        
        return sentiment_logits, aspect_logits


# 2. HÀM TÍNH CHỈ SỐ ĐÁNH GIÁ CHUYÊN SÂU CHO ABSA
def compute_aspect_f1(aspect_trues, aspect_preds, aspects_list, polarities_list):
    """
    Tính bộ chỉ số toàn diện cho cả:
    - Bài toán phát hiện khía cạnh (Aspect Detection: Có đề cập hay Không)
    - Bài toán phân loại khía cạnh + cảm xúc (Aspect-Sentiment Pair)
    """
    num_aspects = aspect_trues.shape[1]
    
    y_true_bin = []
    y_pred_bin = []
    pair_names = []
    
    for a_idx in range(num_aspects):
        for p_idx in range(1, len(polarities_list)):  # Bỏ qua 0 ('none')
            true_col = (aspect_trues[:, a_idx] == p_idx).astype(int)
            pred_col = (aspect_preds[:, a_idx] == p_idx).astype(int)
            
            # Chỉ đưa vào đánh giá những nhãn thực sự có mẫu trong tập dữ liệu (support > 0)
            if true_col.sum() > 0:
                pair_names.append(f"{aspects_list[a_idx]}__{polarities_list[p_idx]}")
                y_true_bin.append(true_col)
                y_pred_bin.append(pred_col)
            
    y_true_bin = np.column_stack(y_true_bin)
    y_pred_bin = np.column_stack(y_pred_bin)
    
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    
    return micro_f1, macro_f1, weighted_f1, y_true_bin, y_pred_bin, pair_names


def evaluate_and_report(model, test_loader, device, sentiment_class_names, aspects_list, polarities_list):
    """Chạy đánh giá toàn diện trên tập Test, in báo cáo chi tiết và lưu ra file."""
    print("\n" + "="*65)
    print("BẮT ĐẦU ĐÁNH GIÁ TOÀN DIỆN TRÊN TẬP TEST...")
    print("="*65)
    
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device, weights_only=True))
    model.eval()
    
    test_sent_preds, test_sent_trues = [], []
    test_asp_probs, test_asp_trues = [], []
    test_start_time = time.time()
    
    # Trọng số hiệu chỉnh cân bằng Khen/Chê khi suy luận
    calibration_weights = np.array(CALIBRATION_WEIGHTS, dtype=np.float32)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, sentiment_labels, aspect_labels = [b.to(device) for b in batch]
            sentiment_logits, aspect_logits = model(input_ids, attention_mask)
            
            test_sent_preds.extend(torch.argmax(sentiment_logits, dim=1).cpu().numpy())
            test_sent_trues.extend(sentiment_labels.cpu().numpy())
            
            probs = torch.softmax(aspect_logits, dim=-1).cpu().numpy()
            test_asp_probs.append(probs)
            test_asp_trues.append(aspect_labels.cpu().numpy())
            
    total_test_time = time.time() - test_start_time
    num_test_samples = len(test_sent_trues)
    
    test_asp_trues_np = np.vstack(test_asp_trues)
    raw_probs_np = np.vstack(test_asp_probs)
    
    # Áp dụng hiệu chỉnh xác suất và lấy nhãn có độ tự tin cao nhất
    cal_probs = raw_probs_np * calibration_weights
    cal_probs = cal_probs / cal_probs.sum(axis=-1, keepdims=True)
    test_asp_preds_np = cal_probs.argmax(axis=-1)

    # 1. Tính chỉ số Sắc thái tổng thể (Overall Sentiment)
    sent_acc = accuracy_score(test_sent_trues, test_sent_preds)
    sent_macro_f1 = f1_score(test_sent_trues, test_sent_preds, average='macro', zero_division=0)
    sent_weighted_f1 = f1_score(test_sent_trues, test_sent_preds, average='weighted', zero_division=0)
    sent_report = classification_report(
        test_sent_trues, test_sent_preds, target_names=sentiment_class_names, digits=4, zero_division=0
    )

    # 2. Tính chỉ số Phát hiện khía cạnh (Aspect Detection: Khác 0 là Có đề cập)
    det_true = (test_asp_trues_np != 0).astype(int)
    det_pred = (test_asp_preds_np != 0).astype(int)
    det_prec = precision_score(det_true, det_pred, average='micro', zero_division=0)
    det_rec = recall_score(det_true, det_pred, average='micro', zero_division=0)
    det_micro_f1 = f1_score(det_true, det_pred, average='micro', zero_division=0)
    det_weighted_f1 = f1_score(det_true, det_pred, average='weighted', zero_division=0)

    # 3. Tính chỉ số Khía cạnh + Cảm xúc (End-to-End ABSA)
    micro_f1, macro_f1, weighted_f1, y_true_bin, y_pred_bin, pair_names = compute_aspect_f1(
        test_asp_trues_np, test_asp_preds_np, aspects_list, polarities_list
    )
    absa_prec = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    absa_rec = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    samples_f1 = f1_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)

    # 4. Bảng chi tiết cho các khía cạnh chủ đạo (Support >= 10 mẫu trong tập Test)
    frequent_mask = y_true_bin.sum(axis=0) >= 10
    frequent_names = [pair_names[i] for i in np.where(frequent_mask)[0]]
    frequent_report = classification_report(
        y_true_bin[:, frequent_mask],
        y_pred_bin[:, frequent_mask],
        target_names=frequent_names,
        digits=4,
        zero_division=0
    )

    # Xây dựng nội dung báo cáo tổng hợp
    report_lines = [
        f"Thời gian dự đoán (Inference Time): {total_test_time:.4f}s / {num_test_samples} mẫu "
        f"(Trung bình: {total_test_time/num_test_samples*1000:.2f} ms/mẫu)",
        "\n" + "="*65,
        "BẢNG 1: ĐÁNH GIÁ PHÂN LOẠI SẮC THÁI TỔNG THỂ (OVERALL SENTIMENT)",
        "="*65,
        f"  - Accuracy (Độ chính xác toàn cục) : {sent_acc*100:.2f}%",
        f"  - Weighted F1-Score                : {sent_weighted_f1*100:.2f}%",
        f"  - Macro F1-Score                   : {sent_macro_f1*100:.2f}%\n",
        sent_report,
        "="*65,
        "BẢNG 2: TỔNG HỢP ĐÁNH GIÁ ĐA KHÍA CẠNH (MULTI-ASPECT ABSA)",
        "="*65,
        "2.1. Nhiệm vụ Phát hiện khía cạnh (Aspect Category Detection - ACD):",
        f"  - Micro Precision                  : {det_prec*100:.2f}%",
        f"  - Micro Recall                     : {det_rec*100:.2f}%",
        f"  - Micro F1-Score                   : {det_micro_f1*100:.2f}%",
        f"  - Weighted F1-Score                : {det_weighted_f1*100:.2f}%",
        "\n2.2. Nhiệm vụ Khía cạnh + Cảm xúc toàn phần (End-to-End ABSA):",
        f"  - Micro Precision                  : {absa_prec*100:.2f}%",
        f"  - Micro Recall                     : {absa_rec*100:.2f}%",
        f"  - Micro F1-Score                   : {micro_f1*100:.2f}%",
        f"  - Weighted F1-Score                : {weighted_f1*100:.2f}%",
        f"  - Samples F1-Score (Theo từng câu) : {samples_f1*100:.2f}%",
        f"  - Macro F1-Score (Toàn bộ {len(pair_names)} nhãn)  : {macro_f1*100:.2f}%",
        "\n" + "="*65,
        "BẢNG 3: CHI TIẾT CÁC KHÍA CẠNH PHỔ BIẾN (SUPPORT >= 10 TRONG TẬP TEST)",
        "="*65,
        frequent_report
    ]
    
    full_report_text = "\n".join(report_lines)
    print(full_report_text)
    
    # Lưu báo cáo ra file .txt
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = METRICS_DIR / "evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    print(f"Đã lưu toàn bộ báo cáo chi tiết tại: {report_file}")

    # Vẽ Confusion Matrix trên tập Test
    test_conf_matrix = confusion_matrix(test_sent_trues, test_sent_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=sentiment_class_names, yticklabels=sentiment_class_names)
    plt.title('Confusion Matrix - Overall Sentiment (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(METRICS_DIR / 'confusion_matrix.png', dpi=300)
    plt.close()


def plot_training_curves(train_losses, val_losses, val_sent_f1s, val_absa_f1s, best_epoch=None):
    """Vẽ biểu đồ Loss và biểu đồ tiến trình F1-Score qua từng Epoch."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Biểu đồ 1: Loss
    ax1.plot(epochs_range, train_losses, marker='o', label='Train Loss')
    ax1.plot(epochs_range, val_losses, marker='s', label='Validation Loss')
    if best_epoch is not None:
        ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Biểu đồ 2: F1-Score trên tập Validation
    ax2.plot(epochs_range, val_sent_f1s, marker='o', color='green', label='Val Overall Sentiment F1')
    ax2.plot(epochs_range, val_absa_f1s, marker='^', color='purple', label='Val ABSA Micro-F1')
    if best_epoch is not None:
        ax2.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_title('Validation F1-Score Progression')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1-Score')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / 'loss_curve.png', dpi=300)
    plt.close()


def train(eval_only=False):
    if not TOKENIZED_DATA_FILE.exists():
        print(f"Không tìm thấy {TOKENIZED_DATA_FILE}. Chạy prepare_data.py trước.")
        return

    with open(LABEL_CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
    sentiment_class_names = classes_info["overall_sentiments"]
    aspects_list = classes_info["aspects"]
    polarities_list = classes_info.get("aspect_polarities", ["none", "positive", "negative", "neutral"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    data = torch.load(TOKENIZED_DATA_FILE, weights_only=False)
    train_loader = DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data['val'], batch_size=BATCH_SIZE)
    test_loader = DataLoader(data['test'], batch_size=BATCH_SIZE)
    
    num_sentiment_classes = data['num_sentiment_classes']
    num_aspects = data['num_aspects']
    num_polarities = data['num_polarities']

    model = HotelCommentModel(num_sentiment_classes, num_aspects, num_polarities).to(device)

    # Nếu bật cờ --eval-only, bỏ qua bước train và chạy thẳng đánh giá từ phobert_best.pt
    if eval_only:
        if not BEST_MODEL_FILE.exists():
            print(f"Không tìm thấy mô hình tại {BEST_MODEL_FILE} để đánh giá.")
            return
        evaluate_and_report(model, test_loader, device, sentiment_class_names, aspects_list, polarities_list)
        return
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn_sentiment = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Cân bằng trọng số giữa [none, positive, negative, neutral]
    aspect_class_weights = torch.tensor(ASPECT_LOSS_WEIGHTS, dtype=torch.float32).to(device)
    loss_fn_aspect = nn.CrossEntropyLoss(weight=aspect_class_weights)

    best_score = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    val_sent_f1s, val_absa_f1s = [], []

    print(
        f"Bắt đầu huấn luyện (Tối đa {EPOCHS} Epochs | "
        f"Early Stopping Patience = {EARLY_STOPPING_PATIENCE}, Min Delta = {EARLY_STOPPING_DELTA})..."
    )
    train_start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids, attention_mask, sentiment_labels, aspect_labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            sentiment_logits, aspect_logits = model(input_ids, attention_mask)
            
            loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiment_labels)
            loss_aspect = loss_fn_aspect(
                aspect_logits.view(-1, num_polarities),
                aspect_labels.view(-1)
            )
            
            loss = loss_sentiment + 2.0 * loss_aspect
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ================= ĐÁNH GIÁ TRÊN TẬP VALIDATION =================
        model.eval()
        total_val_loss = 0
        val_sent_preds, val_sent_trues = [], []
        val_asp_preds, val_asp_trues = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, sentiment_labels, aspect_labels = [b.to(device) for b in batch]
                sentiment_logits, aspect_logits = model(input_ids, attention_mask)
                
                loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiment_labels)
                loss_aspect = loss_fn_aspect(
                    aspect_logits.view(-1, num_polarities),
                    aspect_labels.view(-1)
                )
                loss = loss_sentiment + 2.0 * loss_aspect
                total_val_loss += loss.item()
                
                val_sent_preds.extend(torch.argmax(sentiment_logits, dim=1).cpu().numpy())
                val_sent_trues.extend(sentiment_labels.cpu().numpy())
                
                val_asp_preds.append(torch.argmax(aspect_logits, dim=-1).cpu().numpy())
                val_asp_trues.append(aspect_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_sent_f1 = f1_score(val_sent_trues, val_sent_preds, average='macro', zero_division=0)
        val_sent_acc = accuracy_score(val_sent_trues, val_sent_preds)
        
        val_asp_trues_np = np.vstack(val_asp_trues)
        val_asp_preds_np = np.vstack(val_asp_preds)
        
        # Nhận đủ 6 giá trị trả về từ hàm compute_aspect_f1
        val_asp_micro_f1, val_asp_macro_f1, val_asp_weighted_f1, _, _, _ = compute_aspect_f1(
            val_asp_trues_np, val_asp_preds_np, aspects_list, polarities_list
        )
        
        val_sent_f1s.append(val_sent_f1)
        val_absa_f1s.append(val_asp_micro_f1)
        
        combined_score = 0.4 * val_sent_f1 + 0.6 * val_asp_micro_f1
        epoch_time = time.time() - epoch_start
        
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} ({epoch_time:.0f}s) | "
            f"Loss [T/V]: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"Sent [Acc/F1]: {val_sent_acc*100:.1f}%/{val_sent_f1*100:.1f}% | "
            f"ABSA [Micro/Weighted]: {val_asp_micro_f1*100:.1f}%/{val_asp_weighted_f1*100:.1f}%"
        )

        # KIỂM TRA LƯU BEST MODEL & EARLY STOPPING TỰ ĐỘNG
        if combined_score > best_score + EARLY_STOPPING_DELTA:
            best_score = combined_score
            best_epoch = epoch + 1
            patience_counter = 0
            BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"  -> Đã lưu Best Model mới tại Epoch {best_epoch} (Combined Score: {best_score*100:.2f}%)")
        else:
            patience_counter += 1
            print(
                f"  ->Điểm Val không cải thiện vượt ngưỡng "
                f"({patience_counter}/{EARLY_STOPPING_PATIENCE} epochs liên tiếp)."
            )
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nKÍCH HOẠT EARLY STOPPING TẠI EPOCH {epoch+1}! "
                    f"Mô hình đã hội tụ và đạt đỉnh tại Epoch {best_epoch} (Best Score: {best_score*100:.2f}%)."
                )
                break

    total_train_time = time.time() - train_start_time
    print(f"\nTỔNG THỜI GIAN HUẤN LUYỆN: {total_train_time:.2f} giây ({total_train_time/60:.1f} phút)")

    plot_training_curves(train_losses, val_losses, val_sent_f1s, val_absa_f1s, best_epoch=best_epoch)
    evaluate_and_report(model, test_loader, device, sentiment_class_names, aspects_list, polarities_list)
    print(f"Đã xuất các biểu đồ và báo cáo ra thư mục: {METRICS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá mô hình ABSA")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Chỉ chạy đánh giá trên tập Test bằng model đã lưu (không train lại)"
    )
    args = parser.parse_args()
    train(eval_only=args.eval_only)
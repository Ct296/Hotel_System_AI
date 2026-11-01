import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from config import (
    TOKENIZED_DATA_FILE, BEST_MODEL_FILE, METRICS_DIR,
    MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, LABEL_CLASSES_FILE
)

# 1. KHỞI TẠO KIẾN TRÚC MÔ HÌNH MULTI-TASK PHOBERT
class HotelCommentModel(nn.Module):
    def __init__(self, num_sentiment_classes, num_aspect_classes):
        super(HotelCommentModel, self).__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.sentiment_head = nn.Linear(self.phobert.config.hidden_size, num_sentiment_classes)
        self.aspect_head = nn.Linear(self.phobert.config.hidden_size, num_aspect_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.drop(pooled_output)
        
        sentiment_logits = self.sentiment_head(pooled_output)
        aspect_logits = self.aspect_head(pooled_output)
        
        return sentiment_logits, aspect_logits

# 2. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ
def plot_metrics(train_losses, val_losses, conf_matrix, class_names):
    """Vẽ biểu đồ Loss và Ma trận nhầm lẫn."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Biểu đồ Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(METRICS_DIR / 'loss_curve.png')
    plt.close()

    # 2. Ma trận nhầm lẫn (Confusion Matrix) trên tập Test
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Overall Sentiment (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(METRICS_DIR / 'confusion_matrix.png')
    plt.close()

def train():
    if not TOKENIZED_DATA_FILE.exists():
        print(f"Không tìm thấy {TOKENIZED_DATA_FILE}. Chạy prepare_data.py trước.")
        return

    # Lấy danh sách tên nhãn để hiển thị báo cáo
    with open(LABEL_CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
    sentiment_class_names = classes_info["overall_sentiments"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    data = torch.load(TOKENIZED_DATA_FILE, weights_only=False)
    train_loader = DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data['val'], batch_size=BATCH_SIZE)
    test_loader = DataLoader(data['test'], batch_size=BATCH_SIZE)
    
    num_sentiment_classes = data['num_sentiment_classes']
    num_aspect_classes = data['num_aspect_classes']

    model = HotelCommentModel(num_sentiment_classes, num_aspect_classes).to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn_sentiment = nn.CrossEntropyLoss()
    loss_fn_aspect = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print("Bắt đầu huấn luyện...")
    train_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids, attention_mask, sentiment_labels, aspect_labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            sentiment_logits, aspect_logits = model(input_ids, attention_mask)
            
            loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiment_labels)
            loss_aspect = loss_fn_aspect(aspect_logits, aspect_labels)
            loss = loss_sentiment + loss_aspect
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ================= ĐÁNH GIÁ TRÊN TẬP VALIDATION =================
        model.eval()
        total_val_loss = 0
        val_sent_preds, val_sent_trues = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, sentiment_labels, aspect_labels = [b.to(device) for b in batch]
                sentiment_logits, aspect_logits = model(input_ids, attention_mask)
                
                loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiment_labels)
                loss_aspect = loss_fn_aspect(aspect_logits, aspect_labels)
                loss = loss_sentiment + loss_aspect
                total_val_loss += loss.item()
                
                preds = torch.argmax(sentiment_logits, dim=1)
                val_sent_preds.extend(preds.cpu().numpy())
                val_sent_trues.extend(sentiment_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_f1 = f1_score(val_sent_trues, val_sent_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1 (Macro): {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print("  -> 💾 Đã lưu mô hình tốt nhất!")

    total_train_time = time.time() - train_start_time
    print(f"\nTỔNG THỜI GIAN HUẤN LUYỆN: {total_train_time:.2f} giây")

    # ================= ĐÁNH GIÁ TRÊN TẬP TEST (THI THẬT) =================
    print("\n" + "="*50)
    print("BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST...")
    
    # Nạp lại trọng số tốt nhất đã lưu
    model.load_state_dict(torch.load(BEST_MODEL_FILE, weights_only=True))
    model.eval()
    
    test_sent_preds, test_sent_trues = [], []
    test_start_time = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, sentiment_labels, aspect_labels = [b.to(device) for b in batch]
            sentiment_logits, _ = model(input_ids, attention_mask)
            
            preds = torch.argmax(sentiment_logits, dim=1)
            test_sent_preds.extend(preds.cpu().numpy())
            test_sent_trues.extend(sentiment_labels.cpu().numpy())
            
    total_test_time = time.time() - test_start_time
    num_test_samples = len(test_sent_trues)
    
    print(f"Thời gian dự đoán (Inference Time): {total_test_time:.4f} giây cho {num_test_samples} mẫu (Trung bình: {total_test_time/num_test_samples:.4f} s/mẫu)")
    
    # Báo cáo tổng hợp Precision, Recall, F1 (Macro & Weighted)
    print("\nBÁO CÁO PHÂN LOẠI (Sắc thái tổng thể):")
    report = classification_report(test_sent_trues, test_sent_preds, target_names=sentiment_class_names)
    print(report)

    # Vẽ biểu đồ Loss và Confusion Matrix trên tập Test
    test_conf_matrix = confusion_matrix(test_sent_trues, test_sent_preds)
    plot_metrics(train_losses, val_losses, test_conf_matrix, sentiment_class_names)
    print(f"Đã xuất biểu đồ Loss và Confusion Matrix ra thư mục {METRICS_DIR}")

if __name__ == "__main__":
    train()
    
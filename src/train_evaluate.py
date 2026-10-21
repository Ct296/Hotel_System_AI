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
    MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE
)

# ==============================================================================
# 1. KHỞI TẠO KIẾN TRÚC MÔ HÌNH MULTI-TASK PHOBERT
# ==============================================================================
class HotelCommentModel(nn.Module):
    def __init__(self, num_sentiment_classes, num_aspect_classes):
        super(HotelCommentModel, self).__init__()
        # Load thân mô hình PhoBERT (Chỉ lấy phần trích xuất đặc trưng, không lấy lớp phân loại có sẵn)
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        
        # Dropout giúp chống Overfitting (Học vẹt)
        self.drop = nn.Dropout(p=0.3)
        
        # Head 1: Phân loại sắc thái tổng thể (Multi-class: Pos, Neg, Neu)
        self.sentiment_head = nn.Linear(self.phobert.config.hidden_size, num_sentiment_classes)
        
        # Head 2: Trích xuất đa nhãn khía cạnh (Multi-label: HOTEL#POS, ROOM#NEG...)
        self.aspect_head = nn.Linear(self.phobert.config.hidden_size, num_aspect_classes)
        
    def forward(self, input_ids, attention_mask):
        # Đưa văn bản qua PhoBERT
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Lấy vector đại diện cho toàn bộ câu (Token <s> đầu tiên của kiến trúc RoBERTa/PhoBERT)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.drop(pooled_output)
        
        # Rẽ nhánh ra 2 bài toán
        sentiment_logits = self.sentiment_head(pooled_output)
        aspect_logits = self.aspect_head(pooled_output)
        
        return sentiment_logits, aspect_logits

# ==============================================================================
# 2. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ==============================================================================
def plot_metrics(train_losses, val_losses, conf_matrix):
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

    # 2. Ma trận nhầm lẫn (Confusion Matrix) cho sắc thái tổng thể
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative', 'Neutral'],
                yticklabels=['Positive', 'Negative', 'Neutral'])
    plt.title('Confusion Matrix - Overall Sentiment')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(METRICS_DIR / 'confusion_matrix.png')
    plt.close()

def train():
    if not TOKENIZED_DATA_FILE.exists():
        print(f"❌ Không tìm thấy {TOKENIZED_DATA_FILE}. Chạy prepare_data.py trước.")
        return

    # Tự động chọn GPU nếu có (khi chạy trên Colab), ngược lại dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang sử dụng thiết bị: {device}")

    # Load dữ liệu đã Tokenize
    data = torch.load(TOKENIZED_DATA_FILE, weights_only=False)
    train_loader = DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data['val'], batch_size=BATCH_SIZE)
    
    num_sentiment_classes = data['num_sentiment_classes']
    num_aspect_classes = data['num_aspect_classes']

    # Khởi tạo mô hình
    model = HotelCommentModel(num_sentiment_classes, num_aspect_classes).to(device)
    
    # Optimizer & Hàm Loss
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # Loss cho phân loại 1 nhãn (Sắc thái chung)
    loss_fn_sentiment = nn.CrossEntropyLoss()
    # Loss cho phân loại đa nhãn (Khía cạnh) - Dùng Binary Cross Entropy độc lập cho từng nhãn
    loss_fn_aspect = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print("🔥 Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids, attention_mask, sentiment_labels, aspect_labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            sentiment_logits, aspect_logits = model(input_ids, attention_mask)
            
            # Tính tổng Loss của cả 2 bài toán
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
                
                # Lưu lại dự đoán sắc thái để tính F1 và vẽ ma trận
                preds = torch.argmax(sentiment_logits, dim=1)
                val_sent_preds.extend(preds.cpu().numpy())
                val_sent_trues.extend(sentiment_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_f1 = f1_score(val_sent_trues, val_sent_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1 (Sentiment): {val_f1:.4f}")

        # Lưu mô hình tốt nhất
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            best_conf_matrix = confusion_matrix(val_sent_trues, val_sent_preds)
            print("💾 Đã lưu mô hình tốt nhất!")

    # Vẽ biểu đồ sau khi train xong
    plot_metrics(train_losses, val_losses, best_conf_matrix)
    print(f"📊 Đã xuất biểu đồ Loss và Confusion Matrix ra thư mục {METRICS_DIR}")

if __name__ == "__main__":
    train()
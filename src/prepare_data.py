import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from config import (
    CLEAN_CSV_FILE, LABEL_CLASSES_FILE, TOKENIZED_DATA_FILE,
    MODEL_NAME, MAX_LEN, RANDOM_STATE, SPLITS_DATA_DIR
)

def load_label_mappings():
    """Tải thông tin cấu trúc nhãn từ JSON để tạo từ điển map."""
    with open(LABEL_CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
        
    # Tạo mapping cho sắc thái tổng thể
    overall_sentiments = classes_info["overall_sentiments"]
    sentiment2id = {label: idx for idx, label in enumerate(overall_sentiments)}
    
    # Tạo mapping cho đa nhãn khía cạnh (Aspect + Sentiment)
    aspect_sentiments = classes_info["aspect_sentiments"]
    aspect2id = {label: idx for idx, label in enumerate(aspect_sentiments)}
    
    return sentiment2id, aspect2id, len(aspect_sentiments)

def encode_multi_label(labels_str, aspect2id, num_classes):
    """Chuyển đổi chuỗi đa nhãn (phân cách bằng |) thành vector Multi-hot (0 và 1)."""
    vector = np.zeros(num_classes, dtype=np.float32)
    if not isinstance(labels_str, str) or not labels_str.strip():
        return vector
        
    labels = labels_str.split('|')
    for label in labels:
        if label in aspect2id:
            vector[aspect2id[label]] = 1.0
    return vector

def prepare_data():
    if not CLEAN_CSV_FILE.exists():
        print(f"❌ Không tìm thấy {CLEAN_CSV_FILE}. Hãy chạy clean_data.py trước.")
        return

    print("⏳ Đang tải dữ liệu và từ điển nhãn...")
    df = pd.read_csv(CLEAN_CSV_FILE)
    sentiment2id, aspect2id, num_aspect_classes = load_label_mappings()
    
    # 1. Word Segmentation (Tách từ ghép tiếng Việt cho PhoBERT)
    print("⏳ Đang tách từ tiếng Việt (Word Segmentation)...")
    df['segmented_text'] = df['clean_text'].apply(lambda x: ViTokenizer.tokenize(str(x)))
    
    # 2. Mã hóa nhãn (Encode Labels)
    print("⏳ Đang mã hóa nhãn (Label Encoding)...")
    # Label 1: Sắc thái tổng thể (1 số nguyên)
    df['sentiment_id'] = df['overall_sentiment'].map(sentiment2id)
    
    # Label 2: Khía cạnh đa nhãn (Mảng multi-hot)
    df['aspect_vector'] = df['aspect_labels'].apply(
        lambda x: encode_multi_label(x, aspect2id, num_aspect_classes)
    )
    
    # 3. Chia tập dữ liệu (Train: 80%, Val: 10%, Test: 10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)
    
    # Lưu bản copy CSV ra để dễ kiểm tra sau này
    train_df.to_csv(SPLITS_DATA_DIR / "train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv(SPLITS_DATA_DIR / "val.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv(SPLITS_DATA_DIR / "test.csv", index=False, encoding='utf-8-sig')

    # 4. Băm văn bản (Tokenization)
    print(f"⏳ Đang tải Tokenizer từ {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_dataframe(dataframe):
        texts = dataframe['segmented_text'].tolist()
        
        # Tokenize hàng loạt
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt' # Trả về Tensor của PyTorch
        )
        
        # Chuyển nhãn thành Tensor
        sentiment_tensors = torch.tensor(dataframe['sentiment_id'].tolist(), dtype=torch.long)
        aspect_tensors = torch.tensor(dataframe['aspect_vector'].tolist(), dtype=torch.float32)
        
        # Đóng gói dữ liệu thành TensorDataset
        return torch.utils.data.TensorDataset(
            encoded['input_ids'],
            encoded['attention_mask'],
            sentiment_tensors,
            aspect_tensors
        )
        
    print("⏳ Đang Tokenize văn bản và ép kiểu Tensor...")
    train_dataset = tokenize_dataframe(train_df)
    val_dataset = tokenize_dataframe(val_df)
    test_dataset = tokenize_dataframe(test_df)
    
    # 5. Lưu toàn bộ file `.pt`
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'num_aspect_classes': num_aspect_classes,
        'num_sentiment_classes': len(sentiment2id)
    }, TOKENIZED_DATA_FILE)
    
    print("✅ Hoàn tất!")
    print(f"📦 Dữ liệu Tensor đã lưu tại: {TOKENIZED_DATA_FILE}")

if __name__ == "__main__":
    prepare_data()
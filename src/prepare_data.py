import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from config import (
    CLEAN_CSV_FILE, LABEL_CLASSES_FILE, TOKENIZED_DATA_FILE,
    MODEL_NAME, MAX_LEN, RANDOM_STATE, SPLITS_DATA_DIR, FEATURES_DIR
)

def load_label_mappings():
    with open(LABEL_CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
        
    overall_sentiments = classes_info["overall_sentiments"]
    sentiment2id = {label: idx for idx, label in enumerate(overall_sentiments)}
    
    aspects = classes_info["aspects"]
    aspect2id = {aspect: idx for idx, aspect in enumerate(aspects)}
    
    polarities = classes_info.get("aspect_polarities", ["none", "positive", "negative", "neutral"])
    polarity2id = {pol: idx for idx, pol in enumerate(polarities)}
    
    return sentiment2id, aspect2id, polarity2id

def encode_aspect_polarities(labels_str, aspect2id, polarity2id):
    """
    Chuyển chuỗi 'HOTEL#COMFORT__positive|SERVICE#GENERAL__negative'
    thành vector kích thước (num_aspects,) với giá trị:
    0: none, 1: positive, 2: negative, 3: neutral
    """
    vector = np.zeros(len(aspect2id), dtype=np.int64) # Mặc định 0 là 'none'
    if not isinstance(labels_str, str) or not labels_str.strip():
        return vector
        
    for item in labels_str.split('|'):
        if '__' in item:
            aspect, pol = item.split('__', 1)
            if aspect in aspect2id and pol in polarity2id:
                vector[aspect2id[aspect]] = polarity2id[pol]
    return vector

def prepare_data():
    if not CLEAN_CSV_FILE.exists():
        print(f"Không tìm thấy file clean - {CLEAN_CSV_FILE}.")
        return

    print("Đang tải dữ liệu và từ điển nhãn...")
    df = pd.read_csv(CLEAN_CSV_FILE)
    sentiment2id, aspect2id, polarity2id = load_label_mappings()
    
    print("Đang tách từ tiếng Việt (Word Segmentation)...")
    df['segmented_text'] = df['clean_text'].apply(lambda x: ViTokenizer.tokenize(str(x)))
    
    print("Đang mã hóa nhãn (Label Encoding)...")
    df['sentiment_id'] = df['overall_sentiment'].map(sentiment2id)
    df['aspect_targets'] = df['aspect_labels'].apply(
        lambda x: encode_aspect_polarities(x, aspect2id, polarity2id)
    )
    
    # Chia tập dữ liệu có phân tầng (stratify) để giữ nguyên tỷ lệ nhãn
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment_id']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df['sentiment_id']
    )
    
    SPLITS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DATA_DIR / "train.csv", index=False, encoding='utf-8-sig')
    val_df.to_csv(SPLITS_DATA_DIR / "val.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv(SPLITS_DATA_DIR / "test.csv", index=False, encoding='utf-8-sig')

    print(f"Đang tải Tokenizer từ {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_dataframe(dataframe):
        texts = dataframe['segmented_text'].tolist()
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt' 
        )
        sentiment_tensors = torch.tensor(dataframe['sentiment_id'].tolist(), dtype=torch.long)
        aspect_tensors = torch.tensor(np.array(dataframe['aspect_targets'].tolist()), dtype=torch.long)
        
        return torch.utils.data.TensorDataset(
            encoded['input_ids'],
            encoded['attention_mask'],
            sentiment_tensors,
            aspect_tensors
        )
        
    print("Đang Tokenize văn bản và đóng gói Tensor...")
    train_dataset = tokenize_dataframe(train_df)
    val_dataset = tokenize_dataframe(val_df)
    test_dataset = tokenize_dataframe(test_df)
    
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'num_aspects': len(aspect2id),
        'num_polarities': len(polarity2id),
        'num_sentiment_classes': len(sentiment2id)
    }, TOKENIZED_DATA_FILE)
    
    print(f"Hoàn tất! Dữ liệu Tensor đã lưu tại: {TOKENIZED_DATA_FILE}")

if __name__ == "__main__":
    prepare_data()
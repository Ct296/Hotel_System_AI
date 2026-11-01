import re
import json
import pandas as pd
from pathlib import Path
from config import RAW_CSV_FILE, CLEAN_CSV_FILE, DATA_DIR

ABBREVIATIONS_FILE = DATA_DIR / "abbreviations.json"

def load_abbreviations():
    if not ABBREVIATIONS_FILE.exists():
        print(f"Không tìm thấy {ABBREVIATIONS_FILE}. Bỏ qua bước chuẩn hóa từ viết tắt.")
        return {}
    with open(ABBREVIATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text, abbrev_dict):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    if abbrev_dict:
        for abbr, full_word in abbrev_dict.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, full_word, text)
        
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_cleaning():
    if not RAW_CSV_FILE.exists():
        print(f"Không tìm thấy dữ liệu thô tại {RAW_CSV_FILE}")
        return

    print("Đang tải dữ liệu và từ điển chuẩn hóa...")
    df = pd.read_csv(RAW_CSV_FILE)
    abbrev_dict = load_abbreviations()
    
    print("Đang làm sạch dữ liệu văn bản...")
    df['clean_text'] = df['text'].apply(lambda x: clean_text(x, abbrev_dict))
    df = df[df['clean_text'].str.strip() != '']
    
    CLEAN_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV_FILE, index=False, encoding='utf-8-sig')
    
    print(f"Đã làm sạch thành công {len(df)} bình luận.")
    print(f"File dữ liệu sạch lưu tại: {CLEAN_CSV_FILE}")

if __name__ == "__main__":
    process_cleaning()
import re
import html
import json
import unicodedata
import pandas as pd
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
    
    # 1. Khắc phục các lỗi HTML entities thiếu dấu chấm phẩy (VD: H&atildey -> Hãy)
    text = re.sub(r'&(atilde|agrave|aacute|acirc|egrave|eacute|ecirc|igrave|iacute|ograve|oacute|ocirc|otilde|ugrave|uacute|yacute)(?![a-zA-Z;])', r'&\1;', text, flags=re.IGNORECASE)
    text = html.unescape(text)
    
    # 2. Chuẩn hóa bảng mã tiếng Việt về chuẩn dựng sẵn NFC
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    
    # 3. Xóa URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 4. Tách các dấu gạch ngang, gạch chéo hoặc dấu câu bị dính liền vào chữ (VD: "sạch sẽ-nhân viên")
    text = re.sub(r'[-/]', ' ', text)
    text = re.sub(r'([.,!?;:()])', r' \1 ', text)
    
    # 5. Rút gọn các ký tự kéo dài (VD: đẹpppp -> đẹp)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 6. Chuẩn hóa từ viết tắt
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
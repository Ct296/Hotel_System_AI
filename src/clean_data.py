import re
import json
import pandas as pd
from pathlib import Path
from config import RAW_CSV_FILE, CLEAN_CSV_FILE, DATA_DIR

ABBREVIATIONS_FILE = DATA_DIR / "abbreviations.json"

def load_abbreviations():
    """Tải danh sách từ viết tắt từ file JSON."""
    if not ABBREVIATIONS_FILE.exists():
        print(f"⚠️ Không tìm thấy {ABBREVIATIONS_FILE}. Bỏ qua bước chuẩn hóa từ viết tắt.")
        return {}
    with open(ABBREVIATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def profanity_filter(text):
    """
    Hàm để trống (placeholder) xử lý từ nhạy cảm.
    Hiện tại trả về nguyên gốc để AI tự học ngữ cảnh.
    """
    # Gắn logic kiểm tra từ chửi thề, tục tĩu tại đây nếu cần thiết
    return text

def clean_text(text, abbrev_dict):
    """Làm sạch văn bản nhưng vẫn giữ nguyên dấu câu cho PhoBERT."""
    if not isinstance(text, str):
        return ""
    
    # 1. Chuyển về chữ thường để chuẩn hóa
    text = text.lower()
    
    # 2. Xóa các URL, link rác
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 3. Thu gọn các ký tự lặp lại quá 2 lần (VD: "ngonnnn" -> "ngonn")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 4. Thay thế từ viết tắt dựa trên ranh giới từ (\b)
    # Tránh việc thay thế nhầm chữ "ko" trong từ "khong"
    for abbr, full_word in abbrev_dict.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full_word, text)
        
    # 5. Lọc từ nhạy cảm
    text = profanity_filter(text)
    
    # 6. Chuẩn hóa khoảng trắng và giữ nguyên dấu câu (! ? , .)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_cleaning():
    if not RAW_CSV_FILE.exists():
        print(f"❌ Không tìm thấy dữ liệu thô tại {RAW_CSV_FILE}")
        return

    print("⏳ Đang tải dữ liệu và từ điển chuẩn hóa...")
    df = pd.read_csv(RAW_CSV_FILE)
    abbrev_dict = load_abbreviations()
    
    print("⏳ Đang làm sạch dữ liệu văn bản...")
    # Khởi tạo cột mới chứa văn bản đã làm sạch
    df['clean_text'] = df['text'].apply(lambda x: clean_text(x, abbrev_dict))
    
    # Loại bỏ các dòng trống sau khi làm sạch
    df = df[df['clean_text'].str.strip() != '']
    
    # Lưu ra file CSV mới
    CLEAN_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV_FILE, index=False, encoding='utf-8-sig')
    
    print(f"✅ Đã làm sạch thành công {len(df)} bình luận.")
    print(f"📁 File dữ liệu sạch lưu tại: {CLEAN_CSV_FILE}")

if __name__ == "__main__":
    process_cleaning()
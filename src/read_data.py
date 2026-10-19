import re
import json
import pandas as pd
from config import DATASET_FILE, RAW_CSV_FILE, LABEL_CLASSES_FILE

def determine_overall_sentiment(sentiments):
    """
    Tự động tính sắc thái tổng thể của câu dựa trên các nhãn khía cạnh.
    Quy tắc: Khen > Chê -> positive | Chê > Khen -> negative | Bằng nhau -> neutral
    """
    if not sentiments:
        return "neutral"
    
    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

def process_dataset():
    # Kiểm tra xem file dataset.txt có tồn tại không
    if not DATASET_FILE.exists():
        print(f"Lỗi: Không tìm thấy file {DATASET_FILE}")
        return

    # Đọc nội dung file
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex 1: Tách từng block bình luận bắt đầu bằng #ID
    entry_pattern = re.compile(r'#(\d+)\s*(.*?)\s*(?=(?:#\d+)|\Z)', re.DOTALL)
    # Regex 2: Tách các nhãn trong ngoặc nhọn {KHÍA_CẠNH, sắc_thái}
    label_pattern = re.compile(r'\{([^,]+),\s*(positive|negative|neutral)\}')

    data = []
    all_aspect_sentiments = set()
    all_aspects = set()

    for match in entry_pattern.finditer(content):
        entry_id = match.group(1)
        block = match.group(2).strip()

        # Tìm toàn bộ nhãn trong block
        labels = label_pattern.findall(block)
        
        # Xóa nhãn khỏi block để lấy phần chữ (text) thuần túy
        text = label_pattern.sub('', block).strip()
        # Chuẩn hóa khoảng trắng thừa và ký tự xuống dòng
        text = re.sub(r'\s+', ' ', text).strip()

        aspect_sentiment_labels = []
        sentiments = []

        for aspect, sentiment in labels:
            aspect = aspect.strip()
            sentiment = sentiment.strip().lower()
            
            # Gộp khía cạnh và sắc thái thành 1 nhãn tổ hợp (VD: HOTEL#GENERAL__positive)
            label_str = f"{aspect}__{sentiment}"
            aspect_sentiment_labels.append(label_str)
            sentiments.append(sentiment)
            
            # Đưa vào set để lọc trùng lặp
            all_aspect_sentiments.add(label_str)
            all_aspects.add(aspect)

        overall_sentiment = determine_overall_sentiment(sentiments)

        data.append({
            "id": entry_id,
            "text": text,
            "overall_sentiment": overall_sentiment,
            "aspect_labels": "|".join(aspect_sentiment_labels) # Nối các nhãn bằng dấu |
        })

    # 1. Lưu DataFrame ra file CSV
    df = pd.DataFrame(data)
    # Đảm bảo thư mục tồn tại trước khi lưu
    RAW_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV_FILE, index=False, encoding='utf-8-sig')

    # 2. Lưu danh sách các nhãn quét được ra file JSON
    classes_info = {
        "overall_sentiments": ["positive", "negative", "neutral"],
        "aspects": sorted(list(all_aspects)),
        "aspect_sentiments": sorted(list(all_aspect_sentiments))
    }
    
    LABEL_CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_CLASSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(classes_info, f, ensure_ascii=False, indent=4)

    # In báo cáo ra terminal
    print(f"✅ Đã xử lý thành công {len(df)} bình luận.")
    print(f"📊 Dữ liệu thô lưu tại: {RAW_CSV_FILE}")
    print(f"🏷️ File cấu trúc nhãn lưu tại: {LABEL_CLASSES_FILE}")
    print(f"🔎 Nhận diện tự động: {len(all_aspects)} khía cạnh gốc và {len(all_aspect_sentiments)} tổ hợp nhãn đa khía cạnh.")

if __name__ == "__main__":
    process_dataset()
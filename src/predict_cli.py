import torch
import json
import argparse
from transformers import AutoTokenizer
from pyvi import ViTokenizer

from train_evaluate import HotelCommentModel
from clean_data import clean_text, load_abbreviations
from config import (
    BEST_MODEL_FILE, LABEL_CLASSES_FILE, MODEL_NAME, MAX_LEN
)

def load_environment():
    """Nạp các tài nguyên: Mapping nhãn, Tokenizer và Mô hình."""
    if not BEST_MODEL_FILE.exists() or not LABEL_CLASSES_FILE.exists():
        print(f"❌ Thiếu file trọng số mô hình hoặc file nhãn. Vui lòng train mô hình trước.")
        exit(1)

    # 1. Load từ điển nhãn
    with open(LABEL_CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
        
    id2sentiment = {idx: label for idx, label in enumerate(classes_info["overall_sentiments"])}
    id2aspect = {idx: label for idx, label in enumerate(classes_info["aspect_sentiments"])}
    
    # 2. Load Tokenizer & Model
    print("⏳ Đang khởi động AI (Nạp Tokenizer và Model)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = HotelCommentModel(len(id2sentiment), len(id2aspect))
    # Chuyển weights về thiết bị hiện tại (hỗ trợ load model train bằng GPU sang máy CPU)
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device))
    model.to(device)
    model.eval() # Chuyển sang chế độ suy luận
    
    abbrev_dict = load_abbreviations()
    
    return model, tokenizer, device, id2sentiment, id2aspect, abbrev_dict

def predict(text, model, tokenizer, device, id2sentiment, id2aspect, abbrev_dict):
    """Quy trình dự đoán 1 câu bình luận."""
    # 1. Tiền xử lý
    cleaned = clean_text(text, abbrev_dict)
    segmented = ViTokenizer.tokenize(cleaned)
    
    # 2. Tokenize
    encoded = tokenizer(
        segmented,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # 3. Suy luận
    with torch.no_grad():
        sentiment_logits, aspect_logits = model(input_ids, attention_mask)
        
        # Xử lý kết quả Sentiment (Chọn class có xác suất cao nhất)
        sentiment_pred_id = torch.argmax(sentiment_logits, dim=1).item()
        final_sentiment = id2sentiment[sentiment_pred_id]
        
        # Xử lý kết quả Khía cạnh (Dùng Sigmoid và đặt ngưỡng Threshold = 0.5)
        aspect_probs = torch.sigmoid(aspect_logits).squeeze().cpu().numpy()
        
        # Bắt tất cả các nhãn có xác suất > 50%
        detected_aspects = []
        for idx, prob in enumerate(aspect_probs):
            if prob > 0.2:
                detected_aspects.append((id2aspect[idx], prob))
                
    return cleaned, final_sentiment, detected_aspects

def main():
    parser = argparse.ArgumentParser(description="Test AI phân tích bình luận khách sạn")
    parser.add_argument("--text", type=str, help="Câu bình luận cần phân tích ngay (Bỏ qua chế độ chat)")
    args = parser.parse_args()

    model, tokenizer, device, id2sentiment, id2aspect, abbrev_dict = load_environment()
    
    print("\n" + "="*50)
    print("✨ HỆ THỐNG AI ĐÃ SẴN SÀNG ✨")
    print("Mô hình đang chạy trên:", "GPU 🚀" if device.type == 'cuda' else "CPU 🐢")
    print("Nhập 'q' hoặc 'quit' để thoát.")
    print("="*50 + "\n")

    # Nếu người dùng truyền param --text từ terminal, chạy 1 lần rồi thoát
    if args.text:
        cleaned, sentiment, aspects = predict(args.text, model, tokenizer, device, id2sentiment, id2aspect, abbrev_dict)
        print(f"📝 Văn bản gốc : {args.text}")
        print(f"🧹 Đã làm sạch : {cleaned}")
        print(f"🎯 Sắc thái chung: {sentiment.upper()}")
        print(f"🔍 Khía cạnh chi tiết:")
        if not aspects:
            print("  -> Không nhận diện được khía cạnh cụ thể.")
        for asp, prob in aspects:
            print(f"  -> {asp} (Độ tự tin: {prob*100:.1f}%)")
        return

    # Chế độ tương tác liên tục
    while True:
        try:
            text = input("\nNhập bình luận của khách: ")
            if text.lower() in ['q', 'quit', 'exit']:
                print("Đã thoát chương trình.")
                break
            if not text.strip():
                continue
                
            cleaned, sentiment, aspects = predict(text, model, tokenizer, device, id2sentiment, id2aspect, abbrev_dict)
            
            print(f"  => 🎯 Sắc thái chung: [{sentiment.upper()}]")
            print(f"  => 🔍 Chi tiết:")
            if not aspects:
                print("     (Không nhận diện được khía cạnh)")
            for asp, prob in aspects:
                print(f"     - {asp} ({prob*100:.1f}%)")
                
        except KeyboardInterrupt:
            print("\nĐã thoát chương trình.")
            break

if __name__ == "__main__":
    main()
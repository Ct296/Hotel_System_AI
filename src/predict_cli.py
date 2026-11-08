import re
import torch
import json
import argparse
import numpy as np
from transformers import AutoTokenizer
from pyvi import ViTokenizer

from train_evaluate import HotelCommentModel
from clean_data import clean_text, load_abbreviations
from config import (
    BEST_MODEL_FILE, LABEL_CLASSES_FILE, MODEL_NAME, MAX_LEN, CALIBRATION_WEIGHTS
)

def load_environment():
    """Nạp các tài nguyên: Mapping nhãn, Tokenizer và Mô hình."""
    if not BEST_MODEL_FILE.exists() or not LABEL_CLASSES_FILE.exists():
        raise FileNotFoundError("Thiếu file trọng số mô hình hoặc file nhãn. Vui lòng train mô hình trước.")

    with open(LABEL_CLASSES_FILE, 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
        
    id2sentiment = {idx: label for idx, label in enumerate(classes_info["overall_sentiments"])}
    id2aspect = {idx: label for idx, label in enumerate(classes_info["aspects"])}
    polarities = classes_info.get("aspect_polarities", ["none", "positive", "negative", "neutral"])
    id2polarity = {idx: pol for idx, pol in enumerate(polarities)}
    
    print("⏳ Đang khởi động AI (Nạp Tokenizer và Model)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = HotelCommentModel(len(id2sentiment), len(id2aspect), len(id2polarity))
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    abbrev_dict = load_abbreviations()
    
    return model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict

def split_clauses(cleaned_text):
    """
    Tách câu bình luận phức hợp thành các mệnh đề con tại các từ nối đối lập
    (nhưng, tuy nhiên, mặc dù vậy, điểm trừ, có điều...) hoặc dấu chấm câu
    để tránh hiện tượng vế Khen và vế Chê làm nhiễu chéo nhau.
    """
    pattern = r'\b(?:nhưng mà|nhưng|tuy nhiên|có điều|điểm trừ là|điểm trừ|ngoại trừ|trừ việc)\b|[.!?;]'
    parts = [p.strip(' ,') for p in re.split(pattern, cleaned_text) if len(p.strip(' ,')) >= 4]
    if len(parts) > 1:
        return parts
    return [cleaned_text]

def predict(text, model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict):
    """Quy trình dự đoán bình luận hỗ trợ phân tách mệnh đề đối lập và hiệu chỉnh độ lệch nhãn."""
    cleaned = clean_text(text, abbrev_dict)
    clauses = split_clauses(cleaned)
    
    # Đưa cả câu gốc (ở vị trí 0) và các mệnh đề con (nếu có) vào chạy 1 lần
    all_texts = [cleaned] + (clauses if len(clauses) > 1 else [])
    segmented_texts = [ViTokenizer.tokenize(t) for t in all_texts]
    
    encoded = tokenizer(
        segmented_texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Vector hiệu chỉnh cân bằng xác suất cho [none, positive, negative, neutral]
    # Khắc phục việc dữ liệu gốc có số câu positive nhiều gấp 6 lần negative
    calibration_weights = np.array(CALIBRATION_WEIGHTS, dtype=np.float32)
    
    with torch.no_grad():
        sentiment_logits, aspect_logits = model(input_ids, attention_mask)
        
        # 1. Xác suất khía cạnh trên từng đoạn: shape (num_segments, num_aspects, 4)
        raw_probs = torch.softmax(aspect_logits, dim=-1).cpu().numpy()
        
        # Nếu câu có tách mệnh đề (VD có chữ "nhưng"), ưu tiên lấy kết quả từ các mệnh đề con
        target_probs = raw_probs[1:] if len(all_texts) > 1 else raw_probs
        
        # Hiệu chỉnh cân bằng Khen/Chê và chuẩn hóa lại tổng xác suất = 100%
        cal_probs = target_probs * calibration_weights
        cal_probs = cal_probs / cal_probs.sum(axis=-1, keepdims=True)
        
        detected_dict = {}
        for seg_probs in cal_probs:
            for asp_idx, probs in enumerate(seg_probs):
                pred_pol_id = int(probs.argmax())
                confidence = float(probs[pred_pol_id])
                
                # Chỉ bắt những khía cạnh thực sự rõ ràng (khác 'none' và độ tự tin >= 45%)
                if pred_pol_id != 0 and confidence >= 0.45:
                    asp_name = id2aspect[asp_idx]
                    pol_name = id2polarity[pred_pol_id]
                    # Nếu cùng 1 khía cạnh xuất hiện ở 2 mệnh đề, giữ cái có độ tự tin cao hơn
                    if asp_name not in detected_dict or confidence > detected_dict[asp_name][1]:
                        detected_dict[asp_name] = (pol_name, confidence)
                        
        detected_aspects = [
            (f"{asp}__{pol}", conf) for asp, (pol, conf) in detected_dict.items()
        ]
        detected_aspects.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Xử lý Sắc thái tổng thể (Overall Sentiment) đồng nhất với các khía cạnh tìm được
        pos_cnt = sum(1 for _, (pol, _) in detected_dict.items() if pol == "positive")
        neg_cnt = sum(1 for _, (pol, _) in detected_dict.items() if pol == "negative")
        
        if len(clauses) > 1 and pos_cnt > 0 and neg_cnt > 0:
            if pos_cnt == neg_cnt:
                final_sentiment = "neutral"
            elif neg_cnt > pos_cnt:
                final_sentiment = "negative"
            else:
                final_sentiment = "positive"
        else:
            sentiment_pred_id = torch.argmax(sentiment_logits[0], dim=0).item()
            final_sentiment = id2sentiment[sentiment_pred_id]
                
    return cleaned, final_sentiment, detected_aspects

def main():
    parser = argparse.ArgumentParser(description="Test AI phân tích bình luận khách sạn")
    parser.add_argument("--text", type=str, help="Câu bình luận cần phân tích ngay")
    args = parser.parse_args()

    model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict = load_environment()
    
    print("\n" + "="*50)
    print("✨ HỆ THỐNG AI ĐÃ SẴN SÀNG ✨")
    print("Mô hình đang chạy trên:", "GPU" if device.type == 'cuda' else "CPU 🐢")
    print("Nhập 'q' hoặc 'quit' để thoát.")
    print("="*50 + "\n")

    if args.text:
        cleaned, sentiment, aspects = predict(
            args.text, model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict
        )
        print(f"Văn bản gốc : {args.text}")
        print(f"Đã làm sạch : {cleaned}")
        print(f"Sắc thái chung: {sentiment.upper()}")
        print("Khía cạnh chi tiết:")
        if not aspects:
            print("  -> Không nhận diện được khía cạnh cụ thể.")
        for asp, prob in aspects:
            print(f"  -> {asp} (Độ tự tin: {prob*100:.1f}%)")
        return

    while True:
        try:
            text = input("\nNhập bình luận của khách: ")
            if text.lower() in ['q', 'quit', 'exit']:
                print("Đã thoát chương trình.")
                break
            if not text.strip():
                continue
                
            cleaned, sentiment, aspects = predict(
                text, model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict
            )
            
            print(f"  => 🎯 Sắc thái chung: [{sentiment.upper()}]")
            print("  => 🔍 Chi tiết:")
            if not aspects:
                print("     (Không nhận diện được khía cạnh)")
            for asp, prob in aspects:
                print(f"     - {asp} ({prob*100:.1f}%)")
                
        except KeyboardInterrupt:
            print("\nĐã thoát chương trình.")
            break

if __name__ == "__main__":
    main()
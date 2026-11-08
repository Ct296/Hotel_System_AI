# Trong phần khởi động:
try:
    model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict = load_environment()
    print("API Server: Đã nạp mô hình thành công. Sẵn sàng nhận Request!")
except Exception as e:
    model = None
    logging.error(f"Lỗi nạp mô hình, hãy đảm bảo bạn đã train AI trước: {e}")

# Trong endpoint /api/analyze:
@app.post("/api/analyze", response_model=CommentResponse)
def analyze_comment(request: CommentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản không được để trống")
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình AI chưa được nạp")
    
    try:
        cleaned, sentiment, aspects = predict(
            request.text, model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict
        )
        aspects_list = [asp for asp, prob in aspects]
        return {
            "raw_text": request.text,
            "normalized_text": cleaned,
            "final_label": sentiment,
            "aspect_labels": aspects_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")
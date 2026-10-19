from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# Tái sử dụng trọn vẹn logic nạp mô hình và dự đoán từ predict_cli
from predict_cli import load_environment, predict

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Hotel Comment AI API",
    description="API phân tích cảm xúc và khía cạnh bình luận khách sạn bằng PhoBERT",
    version="1.0.0"
)

# Thiết lập CORS để Vue (nếu gọi trực tiếp) hoặc Spring Boot không bị chặn kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể sửa thành domain của Vue/Springboot khi deploy thực tế
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp AI vào RAM/VRAM 1 lần duy nhất khi server khởi động
print("⏳ Đang khởi động Server và nạp mô hình AI...")
try:
    model, tokenizer, device, id2sentiment, id2aspect, abbrev_dict = load_environment()
    print("✅ API Server: Đã nạp mô hình thành công. Sẵn sàng nhận Request!")
except Exception as e:
    logging.error(f"❌ Lỗi nạp mô hình, hãy đảm bảo bạn đã train AI trước: {e}")

# Định dạng dữ liệu đầu vào (Request Body)
class CommentRequest(BaseModel):
    text: str

# Định dạng dữ liệu đầu ra (Response) cho tài liệu Swagger
class AspectDetail(BaseModel):
    aspect: str
    confidence: float

class CommentResponse(BaseModel):
    original_text: str
    cleaned_text: str
    overall_sentiment: str
    details: list[AspectDetail]


@app.post("/api/analyze", response_model=CommentResponse)
async def analyze_comment(request: CommentRequest):
    """
    Endpoint nhận bình luận và trả về kết quả phân tích.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản không được để trống")
    
    try:
        # Gọi hàm dự đoán
        cleaned, sentiment, aspects = predict(
            request.text, model, tokenizer, device, id2sentiment, id2aspect, abbrev_dict
        )
        
        # Đóng gói mảng khía cạnh
        aspects_list = [
            {"aspect": asp, "confidence": round(float(prob), 4)} 
            for asp, prob in aspects
        ]
        
        return {
            "original_text": request.text,
            "cleaned_text": cleaned,
            "overall_sentiment": sentiment,
            "details": aspects_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")


if __name__ == "__main__":
    # Khởi chạy server tại cổng 8000
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
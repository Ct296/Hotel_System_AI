import sys
from pathlib import Path

# Đảm bảo Python luôn tìm thấy các module trong thư mục src/ dù chạy lệnh từ thư mục gốc
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from predict_cli import load_environment, predict

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Hotel Comment AI API",
    description="API phân tích cảm xúc và khía cạnh bình luận khách sạn bằng PhoBERT",
    version="1.0.0"
)

# Thiết lập CORS để kết nối với Spring Boot / Vue Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp AI vào RAM/VRAM 1 lần duy nhất khi server khởi động
print("Đang khởi động Server và nạp mô hình AI...")
try:
    model, tokenizer, device, id2sentiment, id2aspect, id2polarity, abbrev_dict = load_environment()
    print("API Server: Đã nạp mô hình thành công. Sẵn sàng nhận Request!")
except Exception as e:
    model = None
    logging.error(f"Lỗi nạp mô hình, hãy đảm bảo bạn đã train AI trước: {e}")


# Định dạng dữ liệu đầu vào (Request Body)
class CommentRequest(BaseModel):
    text: str


# Định dạng dữ liệu đầu ra (Response)
class CommentResponse(BaseModel):
    raw_text: str
    normalized_text: str
    final_label: str
    aspect_labels: list[str]


@app.post("/api/analyze", response_model=CommentResponse)
def analyze_comment(request: CommentRequest):
    """
    Endpoint nhận bình luận và trả về kết quả phân tích.
    """
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


if __name__ == "__main__":
    # Khởi chạy server tại cổng 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
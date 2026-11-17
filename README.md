# Hệ Thống AI Phân Tích Bình Luận Khách Sạn (Hotel Comment ABSA)

Dự án ứng dụng mô hình ngôn ngữ tiền huấn luyện **PhoBERT (`vinai/phobert-base-v2`)** kết hợp kiến trúc học đa nhiệm (**Multi-task Learning**) để giải quyết bài toán Phân tích cảm xúc dựa trên khía cạnh (**Aspect-Based Sentiment Analysis - ABSA**) cho các bình luận đánh giá khách sạn bằng tiếng Việt.

---

## 1. Tính Năng Chính

* **Phân tích đa nhiệm (Multi-task Learning):**
  * **Nhánh 1 (Overall Sentiment):** Phân loại sắc thái cảm xúc tổng thể của bình luận (`positive`, `negative`, `neutral`).
  * **Nhánh 2 (Aspect-Based Sentiment):** Nhận diện đồng thời **33 khía cạnh dịch vụ** khách sạn (Phòng ốc, Giá cả, Vị trí, Nhân viên, Tiện ích, Ẩm thực...) cùng sắc thái tương ứng (`none`, `positive`, `negative`, `neutral`).
* **Tiền xử lý văn bản tiếng Việt chuyên sâu:** Tự động sửa lỗi mã HTML entities bị khuyết dấu, chuẩn hóa bảng mã Unicode NFC, chuẩn hóa từ viết tắt/teencode (`ks`, `nv`, `ko`, `dc`...) qua từ điển `abbreviations.json`, và tách từ ghép tiếng Việt bằng `PyVi`.
* **Xử lý câu phức & Hiệu chỉnh xác suất:** Tách mệnh đề đối lập (chứa các từ nối *nhưng, tuy nhiên, có điều, điểm trừ...*) kết hợp trọng số hiệu chỉnh (`CALIBRATION_WEIGHTS`) giúp khắc phục hiện tượng lệch nhãn (Positive Bias).
* **Tích hợp dễ dàng:** Cung cấp cả công cụ dòng lệnh (CLI) để kiểm thử trực tiếp và RESTful API (`FastAPI`) tích hợp sẵn CORS cho hệ thống Backend (Spring Boot) hoặc Frontend (Vue.js).

---

## 2. Cấu Trúc Thư Mục

```text
Hotel_System_AI/
├── artifacts/
│   ├── metrics/                 # Lưu báo cáo kiểm định, đánh giá (.txt) và biểu đồ (.png)
│   └── models/                  # Lưu trọng số mô hình tốt nhất (phobert_best.pt)
├── data/
│   ├── abbreviations.json       # Từ điển chuẩn hóa từ viết tắt tiếng Việt
│   ├── raw/                     # Dữ liệu thô ban đầu (dataset.txt, raw_data.csv)
│   ├── processed/               # Dữ liệu sau khi làm sạch (clean_data.csv)
│   └── splits/                  # Các tập dữ liệu chia theo tỷ lệ 80/10/10 (train/val/test.csv)
├── features/
│   ├── label_classes.json       # Danh sách ánh xạ nhãn cảm xúc và 33 khía cạnh
│   └── tokenized_data.pt        # Dữ liệu đã đóng gói thành Tensor cho PyTorch
├── src/
│   ├── config.py                # Cấu hình đường dẫn hệ thống và siêu tham số (Hyperparameters)
│   ├── check_data.py            # Kiểm định chất lượng dữ liệu và thống kê EDA
│   ├── read_data.py             # Chuyển đổi dữ liệu từ txt sang csv và trích xuất bộ nhãn
│   ├── clean_data.py            # Làm sạch và chuẩn hóa văn bản bình luận
│   ├── prepare_data.py          # Tách từ tiếng Việt, chia tập dữ liệu và đóng gói Tensor
│   ├── train_evaluate.py        # Huấn luyện mô hình Multi-task PhoBERT và xuất báo cáo đánh giá
│   ├── predict_cli.py           # Giao diện dòng lệnh (CLI) dự đoán bình luận trực tiếp
│   └── api_service.py           # Khởi chạy FastAPI Server kết nối với Spring Boot / Vue
├── code.txt                     # Tệp lưu trữ nhanh các câu lệnh chạy hệ thống
└── requirements.txt             # Danh sách các thư viện Python cần thiết
```

---

## 3. Quy Trình Cài Đặt Và Sử Dụng (Kế thừa từ `code.txt`)

### Bước 0: Cài đặt thư viện cần thiết
Yêu cầu hệ thống đã cài đặt Python (khuyến nghị Python 3.9+). Chạy lệnh sau tại thư mục gốc của dự án để cài đặt toàn bộ thư viện:
```bash
pip install -r requirements.txt
```

### Bước 1: Khởi tạo cấu trúc và Kiểm định dữ liệu (EDA)
Khởi tạo cấu trúc đường dẫn (chỉ cần chạy lần đầu):
```bash
python src/config.py
```
Chạy công cụ kiểm định tính toàn vẹn của dữ liệu, thống kê phân bố nhãn, đo độ dài Subword Token để chọn `MAX_LEN` tối ưu và xuất biểu đồ vào `artifacts/metrics/`:
```bash
python src/check_data.py
```

### Bước 2: Xử lý dữ liệu thô
Đọc dữ liệu gốc từ `data/raw/dataset.txt`, bóc tách bình luận cùng các cặp `{KHÍA_CẠNH, cảm_xúc}` sang định dạng bảng `data/raw/raw_data.csv` và tự động trích xuất danh sách nhãn vào `features/label_classes.json`:
```bash
python src/read_data.py
```

### Bước 3: Làm sạch dữ liệu văn bản (`clean_text`)
Xử lý lỗi mã HTML, chuẩn hóa Unicode, loại bỏ ký tự kéo dài, bung từ viết tắt dựa trên `data/abbreviations.json` và lưu kết quả vào `data/processed/clean_data.csv`:
```bash
python src/clean_data.py
```

### Bước 4: Tokenize dữ liệu và đóng gói Tensor
Tách từ tiếng Việt bằng `PyVi`, chia dữ liệu có phân tầng (Stratified Split) theo tỷ lệ 80% Train - 10% Validation - 10% Test, sau đó mã hóa bằng PhoBERT Tokenizer và lưu thành file `features/tokenized_data.pt`:
```bash
python src/prepare_data.py
```

### Bước 5: Huấn luyện và đánh giá mô hình
Huấn luyện mô hình Multi-task PhoBERT (quá trình này tốn nhiều thời gian nhất, khuyến nghị chạy trên máy có GPU CUDA):
```bash
python src/train_evaluate.py
```
* **Lưu ý:** Nếu bạn đã có sẵn file trọng số `artifacts/models/phobert_best.pt` và chỉ muốn chạy lại bước đánh giá trên tập Test mà không cần huấn luyện lại từ đầu, hãy dùng cờ `--eval-only`:
```bash
python src/train_evaluate.py --eval-only
```

### Bước 6: Kiểm thử mô hình trên giao diện Terminal (CLI)

**6.1. Chế độ tương tác liên tục trên Console:**
```bash
python src/predict_cli.py
```
*(Nhập câu bình luận bất kỳ để xem kết quả phân tích, nhập `q` hoặc `quit` để thoát).*

**6.2. Chế độ chạy kiểm thử nhanh 1 câu bằng tham số `--text`:**
```bash
python src/predict_cli.py --text "Khách sạn rộng rãi, sạch nhưng nhân viên phục vụ kém và đồ ăn dở"
```

### Bước 7: Khởi chạy API Server cục bộ
Khởi động máy chủ FastAPI tại cổng `8000` để các ứng dụng tích hợp như **Spring Boot** hoặc **Vue.js** gọi tới:
```bash
python src/api_service.py
```
* **Swagger UI (Tài liệu API tương tác):** Truy cập `http://localhost:8000/docs` sau khi server khởi động.

---

## 4. Đặc Tả API (Dành cho Tích Hợp Hệ Thống)

* **Endpoint:** `POST /api/analyze`
* **URL cục bộ:** `http://localhost:8000/api/analyze`
* **Content-Type:** `application/json`

**Request Body:**
```json
{
  "text": "Khách sạn rộng rãi, sạch nhưng nhân viên phục vụ kém và đồ ăn dở"
}
```

**Response Body (200 OK):**
```json
{
  "raw_text": "Khách sạn rộng rãi, sạch nhưng nhân viên phục vụ kém và đồ ăn dở",
  "normalized_text": "khách sạn rộng rãi , sạch nhưng nhân viên phục vụ kém và đồ ăn dở",
  "final_label": "neutral",
  "aspect_labels": [
    "SERVICE#GENERAL__negative",
    "FOOD&DRINKS#QUALITY__negative",
    "HOTEL#DESIGN&FEATURES__positive",
    "HOTEL#CLEANLINESS__positive"
  ]
}
```

---

## 5. Thông Số Kỹ Thuật & Kết Quả Thực Nghiệm

### Cấu hình huấn luyện (`src/config.py`)
* **Mô hình nền tảng:** `vinai/phobert-base-v2` (Kết hợp `CLS Token` + `Mean Pooling`, kích thước vector đặc trưng 1536 chiều).
* **Độ dài tối đa (`MAX_LEN`):** `256` subword tokens (bao phủ trọn vẹn >99% dữ liệu bình luận thực tế).
* **Batch Size:** `16` | **Learning Rate:** `2e-5` | **Weight Decay:** `0.01`
* **Số vòng lặp tối đa (`EPOCHS`):** `10` (Tích hợp `Early Stopping` với `Patience = 3`).
* **Xử lý mất cân bằng dữ liệu:** Áp dụng trọng số mất mát `ASPECT_LOSS_WEIGHTS = [0.30, 0.90, 1.40, 1.10]` khi huấn luyện và trọng số hậu nghiệm `CALIBRATION_WEIGHTS = [1.00, 0.95, 1.30, 1.10]` khi suy luận.

### Kết quả đánh giá trên tập Test (300 mẫu)
* **Phân loại cảm xúc tổng thể (Overall Sentiment):**
  * **Accuracy:** `90.33%`
  * **Weighted F1-Score:** `90.77%`
  * **Macro F1-Score:** `75.70%`
* **Phát hiện khía cạnh (Aspect Category Detection - ACD):**
  * **Micro F1-Score:** `70.32%` (Micro Recall đạt `78.53%`)
* **Phân tích Khía cạnh + Cảm xúc toàn phần (End-to-End ABSA):**
  * **Micro F1-Score:** `63.08%`
  * **Samples F1-Score:** `63.35%`
* **Tốc độ suy luận trung bình:** `~151.28 ms / bình luận`
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

## 2. Thông Tin Tập Dữ Liệu & Số Lượng Lớp (Dataset)

Tập dữ liệu gồm **3.000 mẫu bình luận** đánh giá khách sạn bằng tiếng Việt (tổng cộng 13.949 lượt gán nhãn khía cạnh), được chia theo tỷ lệ **80% Train / 10% Validation / 10% Test** (300 mẫu kiểm thử). Số lượng lớp đầu ra của hệ thống được chốt theo 2 nhánh bài toán như sau:

* **Nhánh 1 – Sắc thái tổng thể (Overall Sentiment):** Gồm **3 lớp** (`positive`, `negative`, `neutral`).
* **Nhánh 2 – Đa khía cạnh (Multi-Aspect ABSA):** Gồm **33 khía cạnh**, mỗi khía cạnh được phân loại vào **4 lớp trạng thái** (`none` – không đề cập, `positive`, `negative`, `neutral`).
  * *Ghi chú thống kê tổ hợp nhãn:* Theo lý thuyết có tối đa **99 tổ hợp** (33 khía cạnh × 3 sắc thái), nhưng thực tế trên toàn bộ 3.000 mẫu chỉ xuất hiện **92 tổ hợp** (và **76 tổ hợp** trong tập Test). Để mô hình có thể dự đoán độc lập mọi khía cạnh mà không bị bó buộc vào các tổ hợp cứng, kiến trúc nhánh này chọn đầu ra là **33 × 4 trạng thái** (tương ứng 33 bài toán phân loại 4 lớp chạy song song).

---

## 3. Cấu Trúc Thư Mục

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

## 4. Quy Trình Cài Đặt Và Sử Dụng (Kế thừa từ `code.txt`)

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
Chạy công cụ kiểm định tính toàn vẹn của dữ liệu, thống kê số lượng lớp, phân bố nhãn, đo độ dài Subword Token để chọn `MAX_LEN` tối ưu và xuất biểu đồ vào `artifacts/metrics/`:
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

### Bước 4: Tokenize dữ liệu & Chuẩn bị đầu vào cho PhoBERT
Tách từ ghép tiếng Việt bằng `PyVi`, chia dữ liệu có phân tầng (Stratified Split) theo tỷ lệ 80% Train - 10% Validation - 10% Test, sau đó mã hóa văn bản bằng **PhoBERT Tokenizer (`vinai/phobert-base-v2`)** và đóng gói thành Tensor tại `features/tokenized_data.pt`:
```bash
python src/prepare_data.py
```

### Bước 5: Rút trích đặc trưng, Huấn luyện và Đánh giá mô hình

Chạy quy trình rút trích đặc trưng bằng **PhoBERT** và huấn luyện mô hình Multi-task (khuyến nghị chạy trên máy có GPU CUDA):

* **Kiến trúc mô hình sử dụng (2 thành phần chính):**
  1. **Bộ rút trích đặc trưng – PhoBERT (`vinai/phobert-base-v2`):** Tiếp nhận chuỗi token đã tách từ, tinh chỉnh (fine-tune) trọng số để biểu diễn ngữ cảnh câu và gộp `CLS Token` + `Mean Pooling` thành vector đặc trưng **1536 chiều** (`pooled_output`).
  2. **Bộ phân loại – 2 nhánh mạng nơ-ron truyền thẳng (FFNN) chạy song song:** Cùng nhận đầu vào là vector 1536 chiều từ PhoBERT *(đi qua các lớp tuyến tính, hàm kích hoạt `[GELU / ReLU - điền theo train_evaluate.py]` và `[Dropout]`)* gồm:
     * `sentiment_head`: Dự đoán **3 lớp** sắc thái tổng thể.
     * `aspect_head`: Dự đoán **33 × 4 lớp** trạng thái cho 33 khía cạnh.

```bash
python src/train_evaluate.py
```
* **Luồng huấn luyện & kiểm thử trong `train_evaluate.py`:**
```text
for (tối đa 10 epoch, có dừng sớm - Early Stopping với patience = 3) {
    Train      (Cập nhật đồng thời PhoBERT + 2 nhánh FFNN)
      -> Validation (Kiểm định trên tập Val & lưu Best Model)
}
-> Test (Chạy đánh giá 1 lần duy nhất với Best Model tốt nhất)
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

## 5. Đặc Tả API (Dành cho Tích Hợp Hệ Thống)

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

## 6. Kiến Trúc, Thông Số Kỹ Thuật & Kết Quả Thực Nghiệm

### Cơ chế rút trích đặc trưng & Luồng huấn luyện (`src/train_evaluate.py`)
* **Bước rút trích đặc trưng (Feature Extraction):** Sử dụng mô hình ngôn ngữ tiền huấn luyện **PhoBERT (`vinai/phobert-base-v2`)** để trích xuất biểu diễn ngữ cảnh của câu bình luận, sau đó ghép nối vector `CLS Token` (768 chiều) và `Mean Pooling` (768 chiều) tạo thành vector đặc trưng tổng hợp **1536 chiều (`pooled_output`)**.
* **Cơ chế huấn luyện theo từng Batch:**
  * **Chiều xuôi (`forward`):**
    1. **PhoBERT** chạy trước để rút trích ngữ cảnh và tạo ra vector `pooled_output` 1536 chiều.
    2. **2 nhánh FFNN** (`sentiment_head` và `aspect_head`) hoạt động độc lập và song song trên cùng vector 1536 chiều để dự đoán đồng thời cảm xúc tổng thể và 33 khía cạnh.
  * **Chiều ngược (`backward` & `optimizer.step()`):**
    1. Cộng gộp hàm mất mát đa nhiệm của 2 nhánh: $L_{total} = L_{sentiment} + 2.0 \times L_{aspect}$.
    2. Lan truyền ngược (`loss.backward()`): Đạo hàm được tính ngược từ 2 nhánh FFNN truyền về PhoBERT.
    3. Cập nhật trọng số (`optimizer.step()`): Trọng số của **cả PhoBERT lẫn 2 nhánh FFNN** được cập nhật đồng thời cùng lúc.
* **Vòng lặp huấn luyện & Kiểm thử:** Thực hiện tối đa **10 epochs** tích hợp cơ chế dừng sớm (**Early Stopping**, `patience = 3`). Ở mỗi epoch, mô hình chạy `Train` để cập nhật trọng số, sau đó đánh giá trên tập `Validation` để lưu lại `Best Model` (`phobert_best.pt`). Sau khi kết thúc vòng lặp, hệ thống nạp lại `Best Model` để chạy đánh giá **1 lần duy nhất trên tập `Test`**.

### Cấu hình siêu tham số (`src/config.py`)
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
from pathlib import Path

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (PATHS)
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DATA_DIR = DATA_DIR / "splits"

FEATURES_DIR = PROJECT_ROOT / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Đường dẫn các file dữ liệu và mô hình
DATASET_FILE = RAW_DATA_DIR / "dataset.txt"
RAW_CSV_FILE = RAW_DATA_DIR / "raw_data.csv"
CLEAN_CSV_FILE = PROCESSED_DATA_DIR / "clean_data.csv"
TOKENIZED_DATA_FILE = FEATURES_DIR / "tokenized_data.pt"
BEST_MODEL_FILE = MODELS_DIR / "phobert_best.pt"
LABEL_CLASSES_FILE = FEATURES_DIR / "label_classes.json"


# 2. CẤU HÌNH SIÊU THAM SỐ MÔ HÌNH PHOBERT & HUẤN LUYỆN (HYPERPARAMETERS)
MODEL_NAME = "vinai/phobert-base-v2"

# Theo báo cáo kiểm định dữ liệu (data_audit_report.txt):
# Phân vị 99% (99th percentile) của tập dữ liệu là 248 subword tokens.
# Chọn MAX_LEN = 256 giúp bao phủ trọn vẹn 99.07% bình luận (2972/3000 câu) mà không bị cắt cụt ý.
MAX_LEN = 256

BATCH_SIZE = 16               # Số mẫu xử lý song song trong 1 bước cập nhật đạo hàm
EPOCHS = 10                   # Số vòng lặp huấn luyện tối đa
LEARNING_RATE = 2e-5          # Tốc độ học nhỏ (2e-5) để tinh chỉnh nhẹ PhoBERT, tránh Catastrophic Forgetting
WEIGHT_DECAY = 0.01           # Hệ số phạt chuẩn hóa L2 giúp chống học vẹt (Overfitting)
RANDOM_STATE = 42             # Cố định hạt giống ngẫu nhiên để tái lập kết quả chia tập Train/Val/Test

# Cấu hình Dừng sớm tự động (Early Stopping):
EARLY_STOPPING_PATIENCE = 3   # Tự động dừng nếu sau 3 Epochs liên tiếp điểm Validation không tăng
EARLY_STOPPING_DELTA = 0.001  # Mức tăng tối thiểu (0.1%) để được ghi nhận là mô hình có tiến bộ


# 3. CẤU HÌNH TRỌNG SỐ XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU (CLASS IMBALANCE WEIGHTS)
# Thứ tự 4 trạng thái của mỗi khía cạnh: [0: none, 1: positive, 2: negative, 3: neutral]
# Thống kê thực tế trên 3.000 câu (13.949 lượt nhãn xuất hiện trên 33 khía cạnh):
# - Trung bình 1 câu chỉ nhắc tới 4.65/33 khía cạnh -> 85.9% trạng thái là 'none' (Không đề cập).
# - Trong các khía cạnh được đề cập: Positive chiếm 73.33%, Negative chiếm 22.58%, Neutral chiếm 4.09%.

# 3.1. Trọng số hàm mất mát CrossEntropyLoss trong quá trình Huấn luyện (Training):
# Giảm trọng số lớp 'none' xuống 0.30 để mô hình không lười biếng đoán toàn bộ là 'none'.
# Tăng trọng số lớp 'negative' (1.40) và 'neutral' (1.10) để bù đắp việc ít mẫu hơn 'positive' (0.90).
ASPECT_LOSS_WEIGHTS = [0.30, 0.90, 1.40, 1.10]

# 3.2. Trọng số hiệu chỉnh xác suất khi Dự đoán & Đánh giá (Inference Prior Calibration):
# Do dữ liệu gốc có số lượng nhãn Positive (10.229 lượt) nhiều gấp ~3.25 lần Negative (3.150 lượt),
# mô hình có xu hướng thiên vị dự đoán Positive (Positive Bias).
# Mảng CALIBRATION_WEIGHTS = [none, positive, negative, neutral] dùng để nhân hiệu chỉnh hậu nghiệm:
#   - 1.00 (none)    : Giữ nguyên ngưỡng nền của trạng thái Không đề cập để tránh bắt nhầm khía cạnh rác (giữ Precision cao).
#   - 0.95 (positive): Kìm nhẹ xác suất lớp Khen xuống 5% để khử hiện tượng thiên vị (Positive Bias).
#   - 1.30 (negative): Khuếch đại độ nhạy của lớp Chê thêm 30% để bắt nhạy các phàn nàn của khách hàng.
#   - 1.10 (neutral) : Tăng nhẹ 10% cho lớp Trung tính (vốn chỉ chiếm 4.09% dữ liệu).
CALIBRATION_WEIGHTS = [1.00, 0.95, 1.30, 1.10]
from pathlib import Path

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Dựa trên gốc Hotel_System_AI)
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

# File paths
DATASET_FILE = RAW_DATA_DIR / "dataset.txt"
RAW_CSV_FILE = RAW_DATA_DIR / "raw_data.csv"
CLEAN_CSV_FILE = PROCESSED_DATA_DIR / "clean_data.csv"
TOKENIZED_DATA_FILE = FEATURES_DIR / "tokenized_data.pt"
BEST_MODEL_FILE = MODELS_DIR / "phobert_best.pt"

# Nơi lưu trữ danh sách nhãn tự động quét được từ read_data.py
LABEL_CLASSES_FILE = FEATURES_DIR / "label_classes.json"

# ==============================================================================
# 2. CẤU HÌNH MÔ HÌNH PHOBERT & HUẤN LUYỆN
# ==============================================================================
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LEN = 128          
BATCH_SIZE = 16        
EPOCHS = 10            
LEARNING_RATE = 2e-5   
RANDOM_STATE = 42
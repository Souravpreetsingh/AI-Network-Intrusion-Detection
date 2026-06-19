import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

for d in [DATASET_DIR, MODELS_DIR, REPORTS_DIR, SCREENSHOTS_DIR]:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass

DATASET_PATH = os.path.join(DATASET_DIR, "cicids2017_sample.csv")
SYNTHETIC_DATASET_PATH = os.path.join(DATASET_DIR, "cicids2017_synthetic.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1

# CICIDS2017 column names relevant to our features
FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Protocol",
    "Source Port",
    "Destination Port",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Label"
]

# Simplified feature names used in the prediction form
SIMPLE_FEATURES = [
    "duration",
    "packets",
    "bytes",
    "protocol",
    "src_port",
    "dst_port"
]

PROTOCOL_MAP = {
    "TCP": 6,
    "UDP": 17,
    "ICMP": 1,
    "HTTP": 6,
    "DNS": 17,
    "FTP": 6,
    "SSH": 6
}

MODEL_NAMES = ["KNN", "Decision Tree", "Random Forest"]
MODEL_FILES = {
    "KNN": os.path.join(MODELS_DIR, "knn_model.pkl"),
    "Decision Tree": os.path.join(MODELS_DIR, "dt_model.pkl"),
    "Random Forest": os.path.join(MODELS_DIR, "rf_model.pkl"),
}
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
BEST_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "best_model_info.json")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")

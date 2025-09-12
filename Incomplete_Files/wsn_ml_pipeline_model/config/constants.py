
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
DATA_COLUMNS = ['timestamp', 'rssi', 'lqi']

Q1_PERCENTILE = 0.25
Q3_PERCENTILE = 0.75
IQR_K = 1.5
FRAME_SIZE = 100
OVERLAP = 0.5

RAW_DATA_DIR = 'wsn_ml_pipeline_model/data/raw_organized'
CLEANED_DATA_DIR = 'wsn_ml_pipeline_model/data/cleaned'
PREPROCESSED_DATA_DIR = 'wsn_ml_pipeline_model/data/preprocessed_data/frames_'

LOG_FILE_PATH = 'wsn_ml_pipeline_model/logs'
LOG_FILE = 'app.log'
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
LOG_LEVEL = 'INFO'

# -------------------- Training  --------------------
# loading data
TRAIN_INPUT_DIR  = "wsn_ml_pipeline_model/data/preprocessed_data" # Directory containing .npy files
TRAIN_OUTPUT_DIR = "wsn_ml_pipeline_model/training/train_result"  # save plot and log

# training setup
SCENARIO         = "I"       # "I": classify by environment; "II": classify by receiving node
INPUT_CHANNEL    = "rssi"    # Options: "rssi", "lqi", "both"
MODEL_TYPE       = "resnet"  # Options: "cnn", "resnet"
BATCH_SIZE       = 128
EPOCHS           = 5
LR               = 1e-3
SEED             = 42

# training and testing splits
SEEN_SPLIT       = True   # True: random split (Seen); False: leave-one-env example (Unseen)
TEST_SIZE        = 0.25   # for Seen
HELD_OUT_ENV     = 1      # for Scenario II Unseen
HELD_OUT_NODE    = "C"    # for Scenario I Unseen

# File: wsn_ml_pipeline_model/config/constants.py
# This file contains configuration constants for the WSN ML Pipeline Model project.
# It includes settings for data paths, logging, training parameters, and workflow control.

# -------------------- Data Processing --------------------
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
DATA_COLUMNS = ['timestamp', 'rssi', 'lqi']

Q1_PERCENTILE = 0.25
Q3_PERCENTILE = 0.75
IQR_K = 1.5
FRAME_SIZE = 100
OVERLAP = 0.5

# -------------------- Data Paths --------------------
RAW_DATA_DIR = 'wsn_ml_pipeline_model/data/raw'
CLEANED_DATA_DIR = 'wsn_ml_pipeline_model/data/cleaned'
PREPROCESSED_DATA_DIR = 'wsn_ml_pipeline_model/data/preprocessed_data/frames_'
SAVE_FRAME_CSV = False  # or True 

# -------------------- Logging --------------------
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
MODEL_TYPE       = "cnn"     # Options: "cnn", "resnet"
BATCH_SIZE       = 128
EPOCHS           = 5
LR               = 1e-3
SEED             = 42

# training and testing splits
SEEN_SPLIT       = True   # True: random split (Seen); False: leave-one-env example (Unseen)
TEST_SIZE        = 0.25   # for Seen
HELD_OUT_ENV     = 1      # for Scenario II Unseen
HELD_OUT_NODE    = "C"    # for Scenario I Unseen

# -------------------- Workflow/Experiment Control --------------------
RESUME_TRAINING = False        # If True, resume training from latest checkpoint
PREPROCESSING_ACTIVE = True    # If True, run preprocessing step in workflow
N_TRAIN_RUNS = 1               # Number of times to repeat model training
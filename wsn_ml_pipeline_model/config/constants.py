
# File: wsn_ml_pipeline_model/config/constants.py
# This file contains configuration constants for the WSN ML Pipeline Model project.
# It includes settings for data paths, logging, training parameters, and workflow control.
import re
import torch
import random
import numpy as np
# -------------------- Data Processing --------------------
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
DATA_COLUMNS = ['rssi', 'lqi']

Q1_PERCENTILE = 0.25
Q3_PERCENTILE = 0.75
IQR_K = 1.5
FRAME_SIZE = 100
OVERLAP = 0.5

# -------------------- Data Paths --------------------
RAW_DATA_BASE_DIR = 'wsn_ml_pipeline_model/data'
ENV_FOLDERS = {
    1: "Garden",
    2: "Lake",
    3: "Forest",
    4: "Campus",
    5: "Bridge",
}
CLEANED_DATA_DIR = 'wsn_ml_pipeline_model/data/cleaned'
PREPROCESSED_DATA_DIR = 'wsn_ml_pipeline_model/data/preprocessed_data'
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

# Filename format: frames_<Tx>_<Rx>_<env>.npy
NAME_RE = re.compile(
    r"^frames_(?P<tx>[A-Z])_(?P<rx>[A-Z])_(?P<env>[1-5])\.npy$")

# training setup
SCENARIO         = "I"          # "I": classify by environment; "II": classify by receiving node
INPUT_CHANNEL    = "rssi"       # Options: "rssi", "lqi", "both"
MODEL_TYPE       = "cnn"        # Options: "cnn", "resnet"
BATCH_SIZE       = 128
EPOCHS           = 5
LR               = 1e-3
SEED             = 42

# Use GPU if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set random seeds for reproducibility
random.seed(SEED)               # Python's built-in random module
np.random.seed(SEED)            # NumPy
torch.manual_seed(SEED)         # PyTorch


# training and testing splits
SEEN_SPLIT       = True         # True: random split (Seen); False: leave-one-env example (Unseen)
TEST_SIZE        = 0.25         # for Seen
HELD_OUT_ENV     = 1            # for Scenario II Unseen
HELD_OUT_NODE    = "C"          # for Scenario I Unseen

# -------------------- Workflow/Experiment Control --------------------
RESUME_TRAINING = True          # If True, resume training from latest checkpoint
N_TRAIN_RUNS = 10               # Number of times to repeat model training

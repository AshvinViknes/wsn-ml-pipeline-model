
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
DATA_COLUMNS = ['timestamp', 'rssi', 'lqi']

Q1_PERCENTILE = 0.25
Q3_PERCENTILE = 0.75
IQR_K = 1.5
FRAME_SIZE = 100
OVERLAP = 0.5

RAW_DATA_DIR = 'wsn_ml_pipeline_model/data/raw'
CLEANED_DATA_DIR = 'wsn_ml_pipeline_model/data/cleaned'
PREPROCESSED_DATA_DIR = 'wsn_ml_pipeline_model/data/preprocessed_data/frames_'

LOG_FILE_PATH = 'wsn_ml_pipeline_model/logs'
LOG_FILE = 'app.log'
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
LOG_LEVEL = 'INFO'
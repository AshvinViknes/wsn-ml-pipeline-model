# constants.py
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
DATA_COLUMNS = ["timestamp_ms", "rssi", "lqi"]

# Preprocessing defaults (you can tweak)
FRAME_SIZE = 100       # 100 samples per window
OVERLAP = 0.5          # 50% overlap
IQR_K = 1.5            # IQR multiplier for clipping
Q1_PERCENTILE = 25
Q3_PERCENTILE = 75

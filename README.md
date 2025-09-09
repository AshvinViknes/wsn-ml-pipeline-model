# Wireless Sensing Dataset — Cleaned & Windowed (10 s, 2-ch)

## What this is
Preprocessed time-series windows from a wireless sensor network (nodes A, B, C) collected across five environments: garden, lake, forest, campus, bridge. Raw TXT logs were cleaned and converted into fixed-shape windows ready for modeling.

- Window length: 10 s   Overlap: 50%
- Samples per window: 100 (uniform grid)
- Channels (2-ch): RSSI, LQI
- Per-window file: NumPy .npy shaped (100, 2) → (time, channels)

## Folder contents
- windows/ — all window tensors (*.npy) grouped by source file
- dataset_index.csv — one row per window:
  - window_path — relative path to the .npy
  - tx, rx, env — node IDs and environment tag
  - start_ms, end_ms — UTC millisecond bounds of the window
- manifest.json — run metadata (script version, parameters, counts)
- cleaned/ (optional) — 3-column CSVs: timestamp_ms, rssi, lqi

## Quick start
Python:
    import numpy as np, pandas as pd
    idx = pd.read_csv("dataset_index.csv")
    x = np.load(idx.loc[0, "window_path"])   # shape: (100, 2) = (time, channels)
    # If your model expects (channels, time):
    x = x.T

## How these files were produced (high level)
1) Cleaning: tolerant parsing of TXT logs → normalize time to UTC ms, drop missing/non-finite rows, sort + dedupe, basic sanity checks. Output: *_cleaned.csv with (timestamp_ms, rssi, lqi).
2) Windowing: slice each series into 10 s windows with 5 s hop; resample to 100 points on a uniform grid via linear interpolation. Output: .npy windows + dataset_index.csv.

## Recreate the dataset (optional)
    python clean_and_window.py \
      --raw ./raw_organized \
      --cleaned ./cleaned \
      --out ./processed_10s_2ch \
      --win_ms 10000 --step_ms 5000 --n_samples 100

## Notes
- “2-channel” = two synchronized features per time step (RSSI, LQI).
- The pipeline can be extended to more channels (e.g., derived features or additional sensors).


# File: wsn_ml_pipeline_model/preprocess/preprocessing.py
# UPDATED: time-based framing (10 s windows, 50% overlap) with interpolation to 100 samples,
#          and frames contain ONLY two channels: dRSSI_clean and dLQI_clean (no timestamps).
#          Robust to 'timestamp_ms' (preferred) OR integer 'timestamp' in seconds.
import numpy as np
import pandas as pd
import logging
from constants import FRAME_SIZE, OVERLAP, IQR_K, Q1_PERCENTILE, Q3_PERCENTILE

class DataPreprocessor:
    """
    Preprocess a DataFrame containing RSSI and LQI data:
    - Compute first differences (Δ)
    - Min–max normalize to [0,1]
    - Optional IQR clipping for outliers
    - Segment by **time** into 10-second windows with 50% overlap
    - Interpolate each window to exactly `frame_size` samples
    - Return frames with shape: (num_frames, frame_size, 2)  # features last: [dRSSI_clean, dLQI_clean]
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def _choose_timestamp_ms(self, df: pd.DataFrame) -> pd.Series:
        """Return a millisecond timestamp series, robust to either 'timestamp_ms' or 'timestamp' (seconds)."""
        if 'timestamp_ms' in df.columns:
            ts = pd.to_numeric(df['timestamp_ms'], errors='coerce')
        elif 'timestamp' in df.columns:
            ts = pd.to_numeric(df['timestamp'], errors='coerce')
            # If values look like seconds (smaller than ~10^12), convert to ms
            if ts.dropna().max() < 10**12:
                ts = ts * 1000
        else:
            raise ValueError("Input DataFrame must contain 'timestamp_ms' or 'timestamp' column")
        ts = ts.astype('float64')
        return ts

    def preprocess_dataframe(self, df: pd.DataFrame, frame_size: int = FRAME_SIZE, overlap: float = OVERLAP,
                             iqr_k: float = IQR_K, window_ms: int = 10_000) -> np.ndarray:
        """Preprocess DataFrame and return frames (num_frames, frame_size, 2).
        Args:
            df: DataFrame with columns ['timestamp_ms' or 'timestamp', 'rssi', 'lqi']
            frame_size: number of samples per 10-second window (e.g., 100)
            overlap: fractional overlap (e.g., 0.5)
            iqr_k: multiplier for IQR clipping (outlier mitigation)
            window_ms: window length in milliseconds (default 10,000 ms)
        Returns:
            np.ndarray of shape (num_frames, frame_size, 2) with features [dRSSI_clean, dLQI_clean].
        """
        df = df.copy()
        # 1) Timestamps (ms)
        df['timestamp_ms_internal'] = self._choose_timestamp_ms(df)
        df = df.dropna(subset=['timestamp_ms_internal', 'rssi', 'lqi'])
        df = df.sort_values('timestamp_ms_internal').reset_index(drop=True)

        # 2) Deltas
        df['dRSSI'] = df['rssi'].diff()
        df['dLQI']  = df['lqi'].diff()
        df = df.dropna(subset=['dRSSI', 'dLQI']).reset_index(drop=True)

        # 3) Min–max normalization per file
        def _minmax(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors='coerce')
            smin, smax = s.min(), s.max()
            if pd.isna(smin) or pd.isna(smax) or smax == smin:
                return pd.Series(np.zeros(len(s), dtype='float64'), index=s.index)
            return (s - smin) / (smax - smin)

        df['dRSSI_norm'] = _minmax(df['dRSSI'])
        df['dLQI_norm']  = _minmax(df['dLQI'])

        # 4) IQR-based clipping (optional but kept for stability)
        def _clip_iqr(series: pd.Series) -> pd.Series:
            q1 = np.percentile(series, Q1_PERCENTILE)
            q3 = np.percentile(series, Q3_PERCENTILE)
            iqr = q3 - q1
            if iqr == 0:
                return series
            low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            return np.clip(series, low, high)

        df['dRSSI_clean'] = _clip_iqr(df['dRSSI_norm'].to_numpy())
        df['dLQI_clean']  = _clip_iqr(df['dLQI_norm'].to_numpy())

        # 5) Time-based framing with interpolation to fixed length
        start_ts = int(df['timestamp_ms_internal'].min())
        end_ts   = int(df['timestamp_ms_internal'].max())
        step_ms  = int(window_ms * (1 - overlap))
        frames   = []

        def _interp_window(ts_w: np.ndarray, y_w: np.ndarray, s: int, e: int) -> np.ndarray:
            # Deduplicate timestamps (keep last occurrence)
            order = np.argsort(ts_w)
            ts_sorted = ts_w[order]
            y_sorted  = y_w[order]
            uniq_ts, idx = np.unique(ts_sorted, return_index=True)
            y_uniq = y_sorted[idx]
            grid = np.linspace(s, e, frame_size, dtype='float64')
            return np.interp(grid, uniq_ts, y_uniq, left=y_uniq[0], right=y_uniq[-1])

        for s in range(start_ts, end_ts - window_ms + 1, step_ms):
            e = s + window_ms
            mask = (df['timestamp_ms_internal'] >= s) & (df['timestamp_ms_internal'] < e)
            dw = df.loc[mask]
            if len(dw) < 2:
                continue
            ts_w = dw['timestamp_ms_internal'].to_numpy(dtype='float64')
            rssi_w = dw['dRSSI_clean'].astype('float64').to_numpy()
            lqi_w  = dw['dLQI_clean'].astype('float64').to_numpy()

            r = _interp_window(ts_w, rssi_w, s, e)
            l = _interp_window(ts_w, lqi_w,  s, e)
            frame = np.stack([r, l], axis=1)  # (frame_size, 2)
            frames.append(frame)

        frames = np.array(frames, dtype='float64')  # (num_frames, frame_size, 2)
        self.logger.info("Created %d frames (shape each: %s)", frames.shape[0], frames.shape[1:])
        return frames

# File: wsn_ml_pipeline_model/preprocess/preprocessing.py
# This script preprocesses a DataFrame containing RSSI and LQI data, calculating deltas, normalizing values, removing outliers, and segmenting the data into overlapping frames suitable for machine learning tasks.
# It includes functions for outlier removal using the IQR method and normalization to the range [0, 1].
import numpy as np
import pandas as pd
import logging
from wsn_ml_pipeline_model.config.constants import FRAME_SIZE, OVERLAP, IQR_K \
    ,Q1_PERCENTILE, Q3_PERCENTILE

class DataPreprocessor:
    """
    Preprocess a DataFrame containing RSSI and LQI data:
    - Calculate deltas
    - Normalize values
    - Remove outliers using IQR method
    - Segment into overlapping frames for ML tasks
    """

    def __init__(self, logger: logging.Logger = None):
        # Setup logger
        self.logger = logger or logging.getLogger(__name__)

    # Preprocess the dataframe to create frames for ML
    # - Calculate deltas for RSSI and LQI
    # - Normalize values to [0,1]
    # - Remove outliers using IQR method
    # - Segment into overlapping frames of specified size
    # - Return numpy array of frames with shape (num_frames, frame_size, 2)
    def preprocess_dataframe(
            self, df: pd.DataFrame, frame_size=FRAME_SIZE, overlap=OVERLAP, iqr_k=IQR_K) -> np.ndarray:
        """
        Preprocesses the input DataFrame by calculating deltas, normalizing, removing outliers, and segmenting into overlapping frames.

        Args:
            df (pd.DataFrame): Input DataFrame with 'rssi' and 'lqi' columns.
            frame_size (int, optional): Size of each frame. Defaults to app config.
            overlap (float, optional): Fractional overlap between frames. Defaults to app config.
            iqr_k (float, optional): IQR multiplier for outlier removal. Defaults to app config.

        Returns:
            np.ndarray: Array of frames with shape (num_frames, frame_size, 2).
        """
        self.logger.info("Starting preprocessing of dataframe with %d rows", len(df))

        # Calculate delta
        df['dRSSI'] = df['rssi'].diff().fillna(0)
        df['dLQI'] = df['lqi'].diff().fillna(0)

        self.logger.info("Calculated deltas for RSSI, LQI, and timestamp")

        df['dRSSI_norm'] = self.normalize(df['dRSSI'])
        df['dLQI_norm'] = self.normalize(df['dLQI'])
        self.logger.info("Normalized dRSSI and dLQI")

        df['dRSSI_clean'] = self.remove_outliers(iqr_k, df['dRSSI_norm'])
        df['dLQI_clean'] = self.remove_outliers(iqr_k, df['dLQI_norm'])
        self.logger.info("Removed outliers from normalized delta values with iqr_k=%f", iqr_k)

        # Frame segmentation with overlap
        step = int(frame_size * (1 - overlap))
        frames = []
        total_samples = len(df)
        self.logger.info("Segmenting data into frames of size %d with overlap %.2f (step size %d)", frame_size, overlap, step)

        for start in range(0, total_samples - frame_size + 1, step):
            frame_timestamp = df['timestamp'].iloc[start:start + frame_size].values
            frame_rssi = df['dRSSI_clean'].iloc[start:start + frame_size].values
            frame_lqi = df['dLQI_clean'].iloc[start:start + frame_size].values
            # Stack timestamp, RSSI and LQI into shape (frame_size, 3)
            frame = np.stack([frame_timestamp, frame_rssi, frame_lqi], axis=1)
            frames.append(frame)

        self.logger.info("Created %d frames from the dataset", len(frames))
        return np.array(frames)  # shape (num_frames, frame_size, 3)

    # Normalize to [0,1]
    def normalize(self, series: pd.Series) -> pd.Series:
        """Normalizes a pandas Series to the range [0, 1].
        Args:
            series (pd.Series): The input data series to normalize.
        Returns:
            pd.Series: The normalized series with values scaled to [0, 1].
        """
        min_val = series.min()
        max_val = series.max()
        self.logger.debug("Normalizing series: min=%.4f, max=%.4f", min_val, max_val)
        
        if min_val == max_val:
            self.logger.warning("Min and max values are equal, returning zero series to avoid division by zero.")
            return pd.Series(0, index=series.index)
        
        return (series - min_val) / (max_val - min_val + 1e-8)
    
    # Remove outliers via IQR
    # - Calculate Q1, Q3, IQR
    # - Define lower and upper bounds
    # - Clip values outside bounds to these limits
    # - Return cleaned series
    def remove_outliers(self, iqr_k: float, series: pd.Series) -> pd.Series: 
        """
        Removes outliers from a pandas Series using the Interquartile Range (IQR) method.
        Values below (Q1 - iqr_k * IQR) are set to the lower bound, and values above (Q3 + iqr_k * IQR) are set to the upper bound.
        Args:
            iqr_k (float): The multiplier for the IQR to determine outlier thresholds.
            series (pd.Series): The input data series from which to remove outliers.
        Returns:
            pd.Series: The series with outliers clipped to the calculated lower and upper bounds.
        """
        Q1 = series.quantile(Q1_PERCENTILE)
        Q3 = series.quantile(Q3_PERCENTILE)
        self.logger.debug("Calculating outlier thresholds: Q1=%.4f, Q3=%.4f", Q1, Q3)
        # Calculate IQR and define lower/upper bounds
        IQR = Q3 - Q1
        self.logger.debug("Calculated IQR=%.4f", IQR)
        # Define lower and upper bounds for outliers
        if IQR == 0:
            self.logger.warning("IQR is zero, no variability in data. Returning original series.")
            return series
        # Calculate lower and upper bounds
        # Use iqr_k to scale the IQR for outlier detection
        self.logger.debug("Using iqr_k=%.2f for outlier detection", iqr_k)
        # Lower bound: Q1 - iqr_k * IQR
        # Upper bound: Q3 + iqr_k * IQR
        # Clip values outside these bounds
        lower = Q1 - iqr_k * IQR
        upper = Q3 + iqr_k * IQR
        self.logger.debug("Outlier thresholds calculated: lower=%.4f, upper=%.4f", lower, upper)
        clipped = series.clip(lower, upper)
        num_outliers = (series < lower).sum() + (series > upper).sum()
        self.logger.info("Outliers clipped: %d values adjusted", num_outliers)
        return clipped

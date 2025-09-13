# File: wsn_ml_pipeline_model/utils/save_utils.py
# This module provides utility functions to save processed frames in both CSV and NumPy formats.
# It includes functions to save each frame as an individual CSV file and to save all frames
# as a single NumPy array. The frames are expected to have a specific structure with
# columns: ['timestamp', 'dRSSI_clean', 'dLQI_clean']
import os
import logging
import numpy as np
import pandas as pd
from wsn_ml_pipeline_model.config.constants import SAVE_FRAME_CSV

class FrameSaver:
    """
    This class provides utility functions to save processed frames in both CSV and NumPy formats.
    It includes methods to save each frame as an individual CSV file and to save all frames
    as a single NumPy array. The frames are expected to have columns:
    ['timestamp', 'dRSSI_clean', 'dLQI_clean']
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def save_frames_as_csv(self, frames: np.ndarray, output_dir: str) -> None:
        """
        Save each frame as an individual CSV file in the given directory.
        Assumes frames shape is (num_frames, frame_size, 3) with columns:
        ['timestamp','dRSSI_clean', 'dLQI_clean']
        """
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Saving frames to {output_dir}")

        for idx, frame in enumerate(frames):
            df = pd.DataFrame(frame, columns=['timestamp','dRSSI_clean', 'dLQI_clean'])
            df['timestamp'] = df['timestamp'].astype(int)
            out_path = os.path.join(output_dir, f'frame_{idx:04d}.csv')
            df.to_csv(out_path, index=False)

        self.logger.info(f"Saved {len(frames)} frames to {output_dir}")

    def save_frames_as_numpy(self, frames: np.ndarray, output_path: str) -> None:
        """
        Save all frames as a single NumPy .npy file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, frames)
        self.logger.info(f"Saved frames to {output_path}")

    def save_frames(self, frames: np.ndarray, csv_dir: str, npy_path: str) -> None:
        """
        Save frames in both CSV and NumPy formats.

        Args:
            frames (np.ndarray): Array of frames with shape (num_frames, frame_size, 3).
            csv_dir (str): Directory to save individual CSV files.
            npy_path (str): Path to save the NumPy array.
        """
        self.logger.info("Saving frames to CSV and NumPy formats")
        if SAVE_FRAME_CSV:
            self.save_frames_as_csv(frames, csv_dir)
        self.save_frames_as_numpy(frames, npy_path)
        self.logger.info("Frames saved successfully")

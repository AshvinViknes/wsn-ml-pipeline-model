# File: wsn_ml_pipeline_model/preprocess/preprocessing_workflow.py
# This script orchestrates the preprocessing of raw sensor data files, including cleaning, normalization,
# outlier removal, and segmentation into frames suitable for machine learning tasks.
# It handles multiple files, saves cleaned data, and generates frames ready for analysis.
# It uses a logging system to track the progress and any issues encountered during processing.
# It is designed to be run as a standalone script or imported as a module in other workflows
# and is structured to allow for easy extension or modification of the preprocessing steps.        

import os
import sys
import glob
import logging
import traceback
import numpy as np
import pandas as pd
from typing import List

from wsn_ml_pipeline_model.utils.save_utils import FrameSaver
from wsn_ml_pipeline_model.config.logger import LoggerConfigurator   
from wsn_ml_pipeline_model.data_cleaner.clean_data import DataCleaner
from wsn_ml_pipeline_model.preprocess.preprocessing import DataPreprocessor
from wsn_ml_pipeline_model.config.constants import (
    FRAME_SIZE,
    OVERLAP,
    IQR_K,
    RAW_DATA_BASE_DIR,
    ENV_FOLDERS,
    PREPROCESSED_DATA_DIR,
)

class PreprocessingWorkflow:
    """
    This class processes raw data files through a complete preprocessing pipeline,
    including cleaning, normalization, outlier removal, and framing for machine learning.
    It handles multiple files, saves cleaned data, and generates frames ready for analysis.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initializes the preprocessing workflow with a logger and sets up the data cleaner and preprocessor.
        Args:
            logger (logging.Logger, optional): Logger instance for logging messages. If None, a default logger is created.
        """
        self.logger = LoggerConfigurator.setup_logging() or logger
        self.cleaner = DataCleaner(logger=self.logger)
        self.preprocessor = DataPreprocessor(logger=self.logger)
        self.frame_saver = FrameSaver(logger=self.logger)

    # -- helper to map new names -> old training tag (frames_<Tx>_<Rx>_<env>.npy)
    def _infer_ids_from_path(self, raw_file: str):
        # env folder name
        env_name = os.path.basename(os.path.dirname(raw_file))
        # invert ENV_FOLDERS to get numeric id
        env_to_id = {v: k for k, v in ENV_FOLDERS.items()}
        env_id = env_to_id.get(env_name)
        if env_id is None:
            raise ValueError(f"Unknown environment folder '{env_name}'")

        # file like N2_to_N1.txt
        stem = os.path.splitext(os.path.basename(raw_file))[0]
        try:
            tx_n, rx_n = stem.split('_to_')
        except ValueError:
            raise ValueError(f"Bad raw filename (expected N?_to_N?.txt): {raw_file}")

        # map N1/N2/N3 -> A/B/C to satisfy existing NAME_RE for training
        n_to_letter = {"N1": "A", "N2": "B", "N3": "C"}
        if tx_n not in n_to_letter or rx_n not in n_to_letter:
            raise ValueError(f"Unknown node code in '{stem}'")
        tx_letter = n_to_letter[tx_n]
        rx_letter = n_to_letter[rx_n]
        return tx_letter, rx_letter, env_id

    def process_file_pipeline(
        self, raw_file: str, cleaned_dir: str, frame_size=FRAME_SIZE, overlap=OVERLAP, iqr_k=IQR_K
    ) -> np.ndarray:
        """
        Processes a raw data file through a complete preprocessing pipeline and returns framed data for machine learning.

        Steps performed:
        1. Cleans the raw text file and saves it as a CSV.
        2. Loads the cleaned CSV into a pandas DataFrame.
        3. Preprocesses the DataFrame and splits it into overlapping frames.
        4. Saves the resulting frames as both CSV files and a NumPy array.

        Args:
            raw_file (str): Path to the raw input text file.
            cleaned_dir (str): Directory where the cleaned CSV will be saved.
            frame_size (int, optional): Number of samples per frame. Defaults to app config.
            overlap (float, optional): Fractional overlap between frames (0 to 1). Defaults to app config.
            iqr_k (float, optional): Multiplier for IQR-based outlier removal. Defaults to app config.

        Returns:
            np.ndarray: Array of preprocessed, framed data ready for machine learning.
        """
        base = os.path.basename(raw_file)
        name, _ = os.path.splitext(base)
        cleaned_path = os.path.join(cleaned_dir, f"{name}_cleaned.csv")

        self.logger.info(f"START: Processing file '{raw_file}'")

        # Step 1: Clean raw file
        try:
            self.logger.debug(f"Step 1: Cleaning raw file -> Saving to '{cleaned_path}'")
            nrows = self.cleaner.clean_file(raw_file, cleaned_path)
            if nrows == 0 or not os.path.exists(cleaned_path):
                self.logger.error(f"No cleaned CSV produced for '{raw_file}'. Skipping this file.")
                return np.empty((0, frame_size, 2))
            self.logger.info(f"Step 1: Cleaning completed for '{raw_file}'")
        except Exception:
            self.logger.error(f"Step 1: Cleaning failed for '{raw_file}'\n{traceback.format_exc()}")
            raise

        # Step 2: Load cleaned CSV
        try:
            self.logger.debug(f"Step 2: Loading cleaned CSV from '{cleaned_path}'")
            df = pd.read_csv(cleaned_path)
            self.logger.info(f"Step 2: Loaded cleaned CSV with shape {df.shape} for '{cleaned_path}'")
        except Exception:
            self.logger.error(f"Step 2: Failed to load cleaned CSV '{cleaned_path}'\n{traceback.format_exc()}")
            raise

        # Step 3: Preprocess and frame
        try:
            self.logger.debug(
                f"Step 3: Preprocessing dataframe and framing data "
                f"(frame_size={frame_size}, overlap={overlap}, iqr_k={iqr_k})"
            )
            frames = self.preprocessor.preprocess_dataframe(df, frame_size, overlap, iqr_k)
            self.logger.info(f"Step 3: Preprocessing complete, generated {frames.shape[0]} frames for '{raw_file}'")
        except Exception:
            self.logger.error(f"Step 3: Preprocessing failed for '{raw_file}'\n{traceback.format_exc()}")
            raise

        # Step 4: Save frames
        try:
            # Save CSVs into the same environment folder as the raw file
            csv_save_dir = os.path.join(cleaned_dir, f'{name}_frames')

            # Produce .npy in the global preprocessed dir with legacy naming
            tx_letter, rx_letter, env_id = self._infer_ids_from_path(raw_file)
            npy_basename = f"frames_{tx_letter}_{rx_letter}_{env_id}.npy"
            npy_save_path = os.path.join(PREPROCESSED_DATA_DIR, npy_basename)

            self.logger.debug(f"Step 4: Saving frames as CSV in '{csv_save_dir}' and as NumPy array to '{npy_save_path}'")
            self.frame_saver.save_frames(frames, csv_save_dir, npy_save_path)
            self.logger.info(f"Step 4: Saved frames → '{npy_save_path}' (Tx={tx_letter}, Rx={rx_letter}, Env={env_id})")
        except Exception:
            self.logger.error(f"Step 4: Failed to save frames for '{raw_file}'\n{traceback.format_exc()}")
            raise

        self.logger.info(f"Generated {frames.shape[0]} frames with shape {frames.shape[1:]} for '{raw_file}'")

        frames = frames.reshape(-1, frame_size, 2)
        self.logger.debug(f"Final framed data shape: {frames.shape}")
        if frames.size == 0:
            self.logger.warning(f"No valid frames generated for '{raw_file}'")
            return np.empty((0, frame_size, 2))

        self.logger.info(f"END: Processing file '{raw_file}' completed successfully")
        self.logger.info("-----------------------------------------------------------------------")
        return frames

    def batch_process_files(self, raw_files: List[str], cleaned_dir: str) -> List[np.ndarray]:
        """
        Processes a list of raw files in batch, applying the complete preprocessing pipeline to each file.

        Args:
            raw_files (List[str]): List of paths to raw input text files.
            cleaned_dir (str): Directory where cleaned CSV files will be saved.

        Returns:
            List[np.ndarray]: List of numpy arrays containing framed data for each processed file.
        """
        self.logger.info(f"Starting batch processing of {len(raw_files)} files")
        all_frames = []
        for raw_file in raw_files:
            try:
                frames = self.process_file_pipeline(raw_file, cleaned_dir)
                all_frames.append(frames)
            except Exception:
                self.logger.error(f"Error processing file '{raw_file}', skipping to next.\n{traceback.format_exc()}")
                continue
        self.logger.info(f"Batch processing completed. Successfully processed {len(all_frames)} files out of {len(raw_files)}")
        return all_frames

    def preprocess_run(self):
        """
        Main entry point to run the preprocessing workflow.
        It initializes the preprocessing pipeline, finds raw files, processes them,
        and saves the cleaned and framed data.
        """
        self.logger.info("Initiated preprocessing workflow........")

        raw_files = []
        for env_name in ENV_FOLDERS.values():
            env_dir = os.path.join(RAW_DATA_BASE_DIR, env_name)
            env_files = glob.glob(os.path.join(env_dir, "*.txt"))
            raw_files.extend(env_files)
            self.logger.info(f"Discovered {len(env_files)} raw files in {env_name}")

        if not raw_files:
            self.logger.error(f"No .txt files found under: '{RAW_DATA_BASE_DIR}'")
            sys.exit(1)

        self.logger.info(f"Found {len(raw_files)} raw files across environments to process..")

        # Write cleaned CSVs into their environment folder
        for raw_file in raw_files:
            env_dir = os.path.dirname(raw_file)
            try:
                self.process_file_pipeline(raw_file, env_dir)
            except Exception:
                self.logger.error(f"Error processing file '{raw_file}', skipping.\n{traceback.format_exc()}")
                continue

        self.logger.info("Preprocessing workflow completed successfully...............")

if __name__ == "__main__":
    workflow = PreprocessingWorkflow()
    workflow.preprocess_run()

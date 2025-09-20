# File: wsn_ml_pipeline_model/training/config.py
# This file defines the configuration dataclass for training the ML model.

from dataclasses import dataclass

@dataclass
class Config:
    """ 
        Configuration for training the ML model.
    """
    TRAIN_INPUT_DIR: str
    TRAIN_OUTPUT_DIR: str
    SCENARIO: str
    SEEN_SPLIT: bool
    HELD_OUT_ENV: int
    HELD_OUT_NODE: str
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    SEED: int
    INPUT_CHANNEL: str
    MODEL_TYPE: str
    TEST_SIZE: float
    RESUME_TRAINING: bool
    N_TRAIN_RUNS: int

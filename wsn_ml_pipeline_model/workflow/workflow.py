# File: wsn_ml_pipeline_model/workflow/workflow.py
# This module orchestrates the end-to-end machine learning workflow,
# including data preprocessing and model training, with options to
# resume training from checkpoints and control preprocessing steps.

from wsn_ml_pipeline_model.config.logger import LoggerConfigurator
from wsn_ml_pipeline_model.training.train_model import WSNPipeline
from wsn_ml_pipeline_model.preprocess.preprocessing_workflow import PreprocessingWorkflow
from wsn_ml_pipeline_model.config.constants import PREPROCESSING_ACTIVE, RESUME_TRAINING, N_TRAIN_RUNS
class MLWorkflow:
    """
        Orchestrates the end-to-end ML pipeline:
        - Data Preprocessing
        - Model Training
    """
    def __init__(self): 
        """
            Initialize the ML workflow with logging, preprocessing, and training components.
        """
        self.logger = LoggerConfigurator.setup_logging()
        self.trainer = WSNPipeline(logger=self.logger)
        self.preprocessor = PreprocessingWorkflow(logger=self.logger)
    
    def run_pipeline(self ,X=None, y=None, class_map=None, 
            L=None, meta=None,resume_latest=RESUME_TRAINING ,
            n_runs=N_TRAIN_RUNS, run_preprocessing=PREPROCESSING_ACTIVE):
        """
            Run the full ML pipeline: preprocessing followed by training.
            If resume_latest is True, it resumes training from the latest checkpoint.
            If run_preprocessing is False, it skips the preprocessing step. This is useful if preprocessing has already been done.
            n_runs specifies how many training runs to perform if resume_latest is True.
            X, y, class_map, L, meta are optional parameters for training data and metadata.

            Note: If run_preprocessing is set to False, ensure that the training data (X, y, etc.) is already preprocessed and available.
            
            Parameters:
                - X: Feature data (optional, can be loaded in preprocessing)
                - y: Labels (optional, can be loaded in preprocessing)
                - class_map: Class mapping (optional, can be loaded in preprocessing)
                - L: Additional data (optional, can be loaded in preprocessing)
                - meta: Metadata (optional, can be loaded in preprocessing)
                - resume_latest: Whether to resume from the latest checkpoint (default: True)
                - n_runs: Number of training runs if resuming (default: 5)
                - run_preprocessing: Whether to run the preprocessing step (default: True)
            
        """
        
        self.logger.info("Starting end-to-end ML pipeline")
        if run_preprocessing:
            self.logger.info("Running preprocessing workflow")
            self.preprocessor.preprocess_run()
            self.logger.info("Preprocessing completed")
        if not resume_latest:
            self.logger.info("Running training workflow")
            self.trainer.run(X, y, class_map, L, meta)
            self.logger.info("Training completed")
        else:
            self.logger.info("Running n training workflow")
            self.trainer.run_multiple(n_runs=n_runs, resume_latest=resume_latest)
            self.logger.info("Training n completed")
        self.logger.info("ML pipeline completed")
        
if __name__ == "__main__":
    """
        Example usage: Run the full ML pipeline with preprocessing and training.

        You can control the workflow behavior using the constants in `config/constants.py`:
            - PREPROCESSING_ACTIVE: Whether to run preprocessing (True/False)
            - RESUME_TRAINING: Whether to resume training from the latest checkpoint (True/False)
            - N_TRAIN_RUNS: Number of training runs (int)

        Typical usage patterns:

        1. To run the full pipeline with preprocessing and training:
               Set PREPROCESSING_ACTIVE = True, RESUME_TRAINING = False, N_TRAIN_RUNS = 1

        2. To run only the training step, resuming from the latest checkpoint:
               Set PREPROCESSING_ACTIVE = False, RESUME_TRAINING = True, N_TRAIN_RUNS = 1

        3. To run only the training step from scratch (no resuming):
               Set PREPROCESSING_ACTIVE = False, RESUME_TRAINING = False, N_TRAIN_RUNS = 1

        4. To run the full pipeline but skip preprocessing (assuming data is already preprocessed):
               Set PREPROCESSING_ACTIVE = False, RESUME_TRAINING = False, N_TRAIN_RUNS = 1

        5. To run multiple training runs, resuming from the latest checkpoint each time:
               Set PREPROCESSING_ACTIVE = False, RESUME_TRAINING = True, N_TRAIN_RUNS = 10

        6. To run the full pipeline with preprocessing and multiple training runs, resuming each time:
               Set PREPROCESSING_ACTIVE = True, RESUME_TRAINING = True, N_TRAIN_RUNS = 10

        7. To run the full pipeline with preprocessing and multiple training runs from scratch each time:
               Set PREPROCESSING_ACTIVE = True, RESUME_TRAINING = False, N_TRAIN_RUNS = 10

        Just set the constants in `config/constants.py` as needed, then run:

            python -m wsn_ml_pipeline_model.workflow.workflow
    """
    
    workflow = MLWorkflow()
    workflow.run_pipeline(
            run_preprocessing=PREPROCESSING_ACTIVE,
            resume_latest=RESUME_TRAINING ,
            n_runs=N_TRAIN_RUNS
        )
# File: wsn_ml_pipeline_model/workflow/workflow.py
# This module orchestrates the end-to-end machine learning workflow,
# including data preprocessing and model training, with options to
# resume training from checkpoints and control preprocessing steps.

from wsn_ml_pipeline_model.config.logger import LoggerConfigurator
from wsn_ml_pipeline_model.training.train_model import WSNPipeline
from wsn_ml_pipeline_model.preprocess.preprocessing_workflow import PreprocessingWorkflow
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
    
    def run_pipeline(self, X=None, y=None, class_map=None, 
            L=None, meta=None):
        """
            Run the full ML pipeline: preprocessing followed by training.
            The training behavior (whether to resume and how many runs) is controlled 
            by the constants in `config/constants.py` (RESUME_TRAINING, N_TRAIN_RUNS).
            X, y, class_map, L, meta are optional parameters for training data and metadata.

            Note: The preprocessing step always runs before training.
            
            Parameters:
                - X: Feature data (optional, can be loaded in preprocessing)
                - y: Labels (optional, can be loaded in preprocessing)
                - class_map: Class mapping (optional, can be loaded in preprocessing)
                - L: Additional data (optional, can be loaded in preprocessing)
                - meta: Metadata (optional, can be loaded in preprocessing)
            
        """
        
        self.logger.info("Starting end-to-end ML pipeline")
        self.logger.info("Running preprocessing workflow")
        self.preprocessor.preprocess_run()
        self.logger.info("Preprocessing completed")
        self.logger.info("Running training workflow")
        # Delegates behavior (resume vs. from-scratch, number of runs) to trainer config.
        self.trainer.run_multiple()
        self.logger.info("Training completed")
        self.logger.info("ML pipeline completed")
        
if __name__ == "__main__":
    """
        Example usage: Run the full ML pipeline with preprocessing and training.

        You can control the workflow behavior using the constants in `config/constants.py`:
            - RESUME_TRAINING: Whether to resume training from the latest checkpoint (True/False)
            - N_TRAIN_RUNS: Number of training runs (int)

        Typical usage patterns:

        1. To run the full pipeline and training from scratch:
               Set RESUME_TRAINING = False, N_TRAIN_RUNS = 1

        2. To run the full pipeline and resume training from the latest checkpoint:
               Set RESUME_TRAINING = True, N_TRAIN_RUNS = 1

        3. To run multiple training runs, resuming from the latest checkpoint each time:
               Set RESUME_TRAINING = True, N_TRAIN_RUNS = 10

        4. To run multiple training runs from scratch each time:
               Set RESUME_TRAINING = False, N_TRAIN_RUNS = 10

        Just set the constants in `config/constants.py` as needed, then run:

            python -m wsn_ml_pipeline_model.workflow.workflow
    """
    
    workflow = MLWorkflow()
    workflow.run_pipeline()
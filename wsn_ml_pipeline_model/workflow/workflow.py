# workflow.py
# This script orchestrates the end-to-end machine learning workflow, integrating data preprocessing and model training

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

    def run_preprocessing(self):
        """
            Run the data preprocessing workflow.
        """
        self.logger.info("Running preprocessing workflow")
        self.preprocessor.preprocess_run()
        self.logger.info("Preprocessing completed")
    
    def run_training(self,X=None, y=None, class_map=None, L=None, meta=None):
        """
            Run the model training workflow.
        """
        self.logger.info("Running training workflow")
        self.trainer.run(X, y, class_map, L, meta)
        self.logger.info("Training completed")
    
    def run_pipeline(self):
        """
            Execute the full ML pipeline: preprocessing followed by training.
        """
        
        self.logger.info("Starting end-to-end ML pipeline")
        self.run_preprocessing()
        self.run_training()
        self.logger.info("ML pipeline completed")
        
if __name__ == "__main__":
    workflow = MLWorkflow()
    workflow.run_pipeline()
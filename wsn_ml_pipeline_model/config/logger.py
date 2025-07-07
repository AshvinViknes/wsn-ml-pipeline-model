# File: wsn_ml_pipeline_model/config/logger.py
# This module configures logging for the application, setting up both console and file handlers.
# It uses a rotating file handler to manage log file size and backups.
# It provides a class method to set up the logger with specified configurations.
import os
import logging
from logging.handlers import RotatingFileHandler
from wsn_ml_pipeline_model.config.constants import LOG_LEVEL, LOG_FILE,\
    LOG_FILE_PATH, LOG_MAX_BYTES, LOG_BACKUP_COUNT

class LoggerConfigurator:
    """
    This class sets up logging for the application, including console and file handlers.
    It uses a rotating file handler to manage log file size and backups.
    """

    @classmethod
    def setup_logging(cls) -> logging.Logger:
        """
        Configures the logging settings for the application.
        Sets up both console and file handlers with a rotating file handler for log management.

        Returns:
            logger (logging.Logger): Configured logger instance.
        """
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(LOG_LEVEL)

        # Clear existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create a formatter for log messages
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Ensure log directory exists
        if not os.path.exists(LOG_FILE_PATH):
            os.makedirs(LOG_FILE_PATH)

        # Rotating file handler
        fh = RotatingFileHandler(
            f"{LOG_FILE_PATH}/{LOG_FILE}",
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Set up log file with timestamp
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d')}.log"
LOG_PATH = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

def get_logger(name=__name__):
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if logger is called multiple times
    if not logger.handlers:
        # File handler with rotation
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=5
        )  # Rotate after 5MB, keep 5 backups
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        )

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger



import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file='logs/trading_project.log', level=logging.INFO):
    """Function to setup as many loggers as you want"""

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=5) # 5 MB per file, 5 backup files
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # Set the lowest level to capture all messages
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Example of how to use it:
# from utils.logger import setup_logger
# logger = setup_logger(__name__)
# logger.info("This is an info message")
# logger.debug("This is a debug message")

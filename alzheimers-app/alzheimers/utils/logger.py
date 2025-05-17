from loguru import logger
import sys
import os

os.makedirs("logs", exist_ok=True)

def setup_logger(log_file="logs/main_log.log", level="INFO"):
    """
    Sets up Loguru logger with both file and console handlers.
    """
    
    logger.remove()  # Remove default logger to prevent duplicate logs
    
    # Add file handler with rotation and retention
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
        level=level.upper()
    )
    
    # Console logging with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level.upper()
    )
    
    return logger

logger = setup_logger()

def get_logger():
    """
    Returns the configured logger instance.
    """
    return logger
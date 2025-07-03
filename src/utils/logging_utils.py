import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Set up logging configuration for a specific module.
    
    Args:
        name: Name of the logger (usually __name__)
        log_dir: Optional directory for log files. If None, only console logging is used.
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'{name.replace(".", "_")}_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # More detailed logging in file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger 


def init_app_logger() -> logging.Logger:
    """Set up and return the logger for the main app."""
    return setup_logger('app', Path('logs/app')) 
"""
Logger configuration module.

Provides centralized logging configuration with both console and file output.
All loggers in the application should use this module for consistent logging.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: Optional[str] = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup and configure a logger with both console and file handlers.

    Args:
        name: Logger name (default: root logger)
        log_level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or __name__)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # Create formatter with standard format
    # Format: [MODULE:LINE - YYYY-MM-DD HH:MM:SS,mmm - LEVEL] - MESSAGE
    formatter = logging.Formatter(
        fmt='[%(name)s:%(lineno)d-%(asctime)s-%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - log to logs/chatbot.log in project root
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "chatbot.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with default configuration.

    Args:
        name: Logger name (default: root logger)
                Usually use __name__ to get module-specific logger

    Returns:
        Configured logger instance
    """
    return setup_logger(name=name)

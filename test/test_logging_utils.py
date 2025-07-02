import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logging_utils import setup_logger

def test_setup_logger_console():
    logger = setup_logger('test_logger')
    assert logger.name == 'test_logger'
    assert any(h for h in logger.handlers if hasattr(h, 'stream'))

def test_setup_logger_with_log_dir(tmp_path):
    logger = setup_logger('test_logger_file', log_dir=tmp_path)
    assert logger.name == 'test_logger_file'
    assert any(h for h in logger.handlers if hasattr(h, 'baseFilename')) 
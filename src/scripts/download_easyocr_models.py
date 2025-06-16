import os
import sys
import logging
from pathlib import Path
import easyocr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_easyocr_models():
    """Download and set up EasyOCR English model."""
    try:
        logger.info("Initializing EasyOCR with English model...")
        reader = easyocr.Reader(['en'])
        
        # Get the model directory
        model_dir = Path.home() / ".cache" / "handwriting_ocr"
        logger.info(f"\nModels are located in: {model_dir}")
        logger.info("\nEnglish model setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during model setup: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_easyocr_models()
    sys.exit(0 if success else 1) 
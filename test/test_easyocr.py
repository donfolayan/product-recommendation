import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project_utils import setup_project_path
setup_project_path()
from src.utils.handwriting_ocr import HandwritingOCR
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent

def test_easyocr():
    logger.info("Initializing HandwritingOCR with CPU mode...")
    ocr = HandwritingOCR(use_gpu=False)

    image_dir = project_root / "test" / "raw_images"
    image_paths = [str(f) for f in image_dir.glob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]]

    if not image_paths:
        logger.warning(f"No image files found in {image_dir}")
        return

    for image_path in image_paths:
        logger.info(f"\nProcessing image: {image_path}")

        result_dict = ocr.process(image_path)

        logger.info("\nDetected text:")
        print(f"Text: {result_dict['text']}")
        print(f"Confidence: {result_dict['confidence']:.2f}")
        print(f"Engine Used: {result_dict['engine_used']}")
        print("---")

if __name__ == "__main__":
    test_easyocr() 
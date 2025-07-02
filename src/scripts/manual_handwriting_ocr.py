import sys
import time
from pathlib import Path
from datetime import datetime
from src.utils.logging_utils import setup_logger

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.handwriting_ocr import HandwritingOCR

# Set up logging using the project utility
logger = setup_logger(__name__, Path('logs'))

def run_ocr_tests(test_dir: Path, output_dir: Path, use_gpu: bool = False):
    """Run OCR tests on all images in the test directory.
    
    Args:
        test_dir: Directory containing test images
        output_dir: Directory to save results
        use_gpu: Whether to use GPU (default: False for CPU)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model directory
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing OCR with CPU optimizations...")
    ocr = HandwritingOCR(use_gpu=False)
    
    # Get all image files
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
    total_images = len(image_files)
    
    logger.info(f"Found {total_images} images to process")
    
    # Process each image
    results = []
    for idx, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"\nProcessing image {idx}/{total_images}: {image_path.name}")
            start_time = time.time()
            
            # Process with both engines
            result = ocr.process(str(image_path))
            processing_time = time.time() - start_time
            
            # Add metadata
            result.update({
                "image": image_path.name,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            })
            
            results.append(result)
            
            # Log results
            logger.info(f"Text: {result['text']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info(f"Engine: {result['engine']}")
            if result['trocr_text']:
                logger.info(f"TrOCR text: {result['trocr_text']}")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
    
    return results

def main():
    """Run OCR tests on all images in the test directory."""
    test_dir = project_root / "test" / "raw_images"
    output_dir = project_root / "test" / "results"
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    run_ocr_tests(test_dir, output_dir)

if __name__ == "__main__":
    main() 
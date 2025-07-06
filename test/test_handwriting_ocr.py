from pathlib import Path
import pytest
from src.utils.project_utils import setup_project_path
setup_project_path()
from src.utils.handwriting_ocr import HandwritingOCR

# Directory with test images
TEST_IMAGE_DIR = Path(__file__).parent.parent / "test" / "raw_images"

# Collect all image files
image_files = list(TEST_IMAGE_DIR.glob("*.jpg")) + list(TEST_IMAGE_DIR.glob("*.jpeg")) + list(TEST_IMAGE_DIR.glob("*.png"))

@pytest.mark.parametrize("image_path", image_files)
def test_handwriting_ocr_on_images(image_path):
    ocr = HandwritingOCR(use_gpu=False)
    result = ocr.process(str(image_path))
    # Basic checks: result should have text and confidence
    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["confidence"], float) 
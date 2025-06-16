import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required Python packages."""
    logger.info("Installing required packages...")
    
    # List of required packages
    packages = [
        'easyocr',
        'torch',
        'torchvision',
        'transformers',
        'Pillow',
        'opencv-python',
        'numpy',
        'pandas'
    ]
    
    try:
        # Install packages
        for package in packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        
        logger.info("All packages installed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {str(e)}")
        sys.exit(1)

def main():
    """Main setup function."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Create scripts directory if it doesn't exist
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    # Install dependencies
    install_dependencies()
    
    # Download EasyOCR models
    logger.info("\nDownloading EasyOCR models...")
    download_script = scripts_dir / "download_easyocr_models.py"
    if download_script.exists():
        subprocess.check_call([sys.executable, str(download_script)])
    else:
        logger.error("Download script not found!")
        sys.exit(1)
    
    logger.info("\nSetup completed successfully!")
    logger.info("You can now run the OCR tests using:")
    logger.info("python test/test_handwriting_ocr.py")

if __name__ == "__main__":
    main() 
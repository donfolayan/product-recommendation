import requests
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_endpoint(image_path):
    """Test the API endpoint with a specific image."""
    url = 'http://localhost:5000/api/v1/detect-product'
    
    try:
        # Open and send the image
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files)
        
        # Log the response
        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response: {response.json()}")
        
        return response.json()
        
    except Exception as e:
        logger.error(f"Error testing API: {str(e)}")
        return None

def main():
    # Test with the same image we used in test_model.py
    image_path = Path(__file__).parent.parent.parent / 'static' / 'images' / '20726' / '20726_1.jpg'
    
    if not image_path.exists():
        logger.error(f"Image not found at {image_path}")
        return
    
    test_api_endpoint(image_path)

if __name__ == "__main__":
    main() 
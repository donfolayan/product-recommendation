import sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from src.utils.model_loader import load_model
from src.services.recommendation_service import RecommendationService
from src.utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, Path('logs'))

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Initialize blueprint
image_bp = Blueprint('image', __name__)

# Initialize model and label mapping
try:
    model, label_mapping = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None
    label_mapping = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@image_bp.route('/detect-product', methods=['POST'])
def detect_product():
    """
    Detect product from image and find similar products.
    
    Returns:
        JSON response containing:
        - detected_class: The class predicted by the CNN
        - confidence: Confidence score of the prediction
        - similar_products: List of similar products with their descriptions
    """
    if model is None or label_mapping is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided'
            }), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                'error': 'No selected file'
            }), 400
            
        # Get top_k parameter
        top_k = request.args.get('top_k', default=5, type=int)
        
        # Read and preprocess image
        img = Image.open(image_file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Get predicted class
        predicted_class = label_mapping[str(predicted.item())]
        confidence_score = confidence.item()
        
        logger.info(f"Detected: {predicted_class} ({confidence_score:.2%})")
        
        # Get similar products using recommendation service
        try:
            # Get recommendation service from current_app
            recommendation_service = current_app.recommendation_service
            
            # Use the predicted class as the query
            products, response = recommendation_service.get_recommendations(
                query=predicted_class,
                top_k=top_k
            )
            
            logger.info(f"Found {len(products)} similar products")
            
            # Format response
            return jsonify({
                'status': 'success',
                'data': {
                    'detected_class': predicted_class,
                    'confidence': confidence_score,
                    'similar_products': products,
                    'response': response
                }
            })
            
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            # Return success without recommendations
            return jsonify({
                'status': 'success',
                'data': {
                    'detected_class': predicted_class,
                    'confidence': confidence_score,
                    'similar_products': [],
                    'response': "Successfully detected product class"
                }
            })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500 
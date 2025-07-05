import sys
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from flask import Blueprint, request, jsonify, current_app, Response
from typing import Union, Tuple
from src.utils.logging_utils import setup_logger
from src.error_handlers import create_error_response
from src.initialization import model, label_mapping, transform

# Ensure log directory exists
log_dir = Path(__file__).resolve().parent.parent.parent / 'logs/image_detection'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(__name__, log_dir)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Initialize blueprint
image_bp = Blueprint('image', __name__)



@image_bp.route('/product-detections', methods=['POST'])
def detect_product() -> Union[Response, Tuple[Response, int]]:
    """
    Detect product from image and find similar products.
    
    Returns:
        JSON response containing:
        - detected_class: The class predicted by the CNN
        - confidence: Confidence score of the prediction
        - similar_products: List of similar products with their descriptions
    """
    if model is None or label_mapping is None:
        return create_error_response('Model not loaded', 500)
    
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return create_error_response('No image file provided', 400)
            
        image_file = request.files['image']
        if image_file.filename == '':
            return create_error_response('No selected file', 400)
            
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
            recommendation_service = getattr(current_app, 'recommendation_service', None)
            if recommendation_service is None:
                logger.warning("Recommendation service not available")
                raise RuntimeError("Recommendation service not initialized")
            
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
        return create_error_response(str(e), 500) 
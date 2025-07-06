import json
import torch
from src.utils.project_utils import setup_project_path

setup_project_path()

from pathlib import Path
from PIL import Image
from flask import Blueprint, request, jsonify, current_app, Response
from typing import Union, Tuple
from src.utils.logging_utils import setup_logger
from src.error_handlers import create_error_response
from src.initialization import model, stock_code_mapping, transform

# Ensure log directory exists
log_dir = Path(__file__).resolve().parent.parent.parent / 'logs/image_detection'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(__name__, log_dir)

image_bp = Blueprint('image', __name__)



@image_bp.route('/product-detections', methods=['POST'])
def detect_product() -> Union[Response, Tuple[Response, int]]:
    """
    Detect product from image and find similar products.
    
    Returns:
        JSON response containing:
        - detected_stock_code: The stock code predicted by the CNN
        - detected_product: The product name for display
        - confidence: Confidence score of the prediction
        - similar_products: List of similar products with their descriptions
    """
    if model is None or stock_code_mapping is None:
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
            
        # Get predicted stock code and product name
        predicted_idx = str(predicted.item())
        predicted_stock_code = stock_code_mapping[predicted_idx]
        
        # Load stock code to product name mapping
        stock_code_to_product_path = Path(__file__).resolve().parent.parent / 'data' / 'dataset' / 'stock_code_to_product.json'
        with open(stock_code_to_product_path, 'r') as f:
            stock_code_to_product = json.load(f)
        
        predicted_product = stock_code_to_product[predicted_stock_code]
        confidence_score = confidence.item()
        
        logger.info(f"Detected: {predicted_product} (StockCode: {predicted_stock_code}) ({confidence_score:.2%})")
        
        # Get similar products using recommendation service
        try:
            # Get recommendation service from current_app
            recommendation_service = getattr(current_app, 'recommendation_service', None)
            if recommendation_service is None:
                logger.warning("Recommendation service not available")
                raise RuntimeError("Recommendation service not initialized")
            
            # Use the predicted product name as the query for recommendations
            products, response = recommendation_service.get_recommendations(
                query=predicted_product,
                top_k=top_k
            )
            
            logger.info(f"Found {len(products)} similar products")
            
            # Format response
            return jsonify({
                'status': 'success',
                'data': {
                    'detected_stock_code': predicted_stock_code,
                    'detected_product': predicted_product,
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
                    'detected_stock_code': predicted_stock_code,
                    'detected_product': predicted_product,
                    'confidence': confidence_score,
                    'similar_products': [],
                    'response': "Successfully detected product class"
                }
            })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return create_error_response(str(e), 500) 
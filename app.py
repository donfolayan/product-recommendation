from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from src.utils.vector_db_utils import VectorDBManager
from src.utils.handwriting_ocr import HandwritingOCR
from src.services.recommendation_service import RecommendationService
import os
from dotenv import load_dotenv
from src.endpoints.image_detection import image_bp, model, label_mapping, transform
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load environment variables
load_dotenv()

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

def create_error_response(message: str, status_code: int = 500, details: dict = None) -> tuple:
    """Create a standardized error response."""
    response = {
        'status': 'error',
        'message': message
    }
    if details:
        response['details'] = details
    return jsonify(response), status_code

# Initialize services
try:
    # Initialize OCR
    ocr = HandwritingOCR(use_gpu=False)
    logger.info("OCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OCR: {str(e)}")
    ocr = None

async def initialize_services():
    """Initialize all required services and models."""
    try:
        # Initialize VectorDBManager
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set!")

        vector_db = VectorDBManager(api_key=pinecone_api_key)
        await asyncio.get_event_loop().run_in_executor(executor, vector_db.initialize)
        
        # Initialize recommendation service with the vector_db's model and index
        recommendation_service = RecommendationService(
            model=vector_db.model,
            index=vector_db.index
        )
        
        return recommendation_service

    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}", exc_info=True)
        raise

# Initialize services
try:
    recommendation_service = asyncio.run(initialize_services())
    # Make recommendation service available to blueprints
    app.recommendation_service = recommendation_service
except Exception as e:
    logger.critical(f"Failed to initialize services: {str(e)}")
    raise

# Register blueprints
app.register_blueprint(image_bp, url_prefix='/api/v1')

@app.route('/')
def home():
    """Home page route."""
    return render_template('index.html')

@app.route('/api/product-recommendation', methods=['POST'])
@async_route
async def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Accepts both JSON and form data.
    """
    try:
        # Get query from either JSON or form data
        if request.is_json:
            data = request.get_json()
            query = data.get('query')
            top_k = data.get('top_k', 5)
        else:
            query = request.form.get('query')
            top_k = int(request.form.get('top_k', 5))

        if not query:
            return create_error_response('Missing required field: query', 400)

        products, response = await recommendation_service.get_recommendations_async(query, top_k)
        
        return jsonify({
            'status': 'success',
            'products': products,
            'response': response
        })

    except Exception as e:
        logger.error(f"Error in product_recommendation endpoint: {str(e)}", exc_info=True)
        return create_error_response('An error occurred while processing your request')

@app.route('/api/ocr-query', methods=['POST'])
@async_route
async def process_ocr_query():
    """
    Endpoint for OCR-based product recommendations.
    Uses HandwritingOCR for text extraction with confidence scores.
    """
    try:
        if ocr is None:
            return create_error_response('OCR service not initialized')

        if 'file' not in request.files:
            return create_error_response('No file provided', 400)
            
        file = request.files['file']
        if file.filename == '':
            return create_error_response('No selected file', 400)
            
        # Save file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.png')
        file.save(temp_path)
        
        try:
            # Process image with HandwritingOCR in thread pool
            ocr_result = await asyncio.get_event_loop().run_in_executor(
                executor,
                ocr.process,
                temp_path
            )
            
            # Get recommendations based on extracted text
            products, response = await recommendation_service.get_recommendations_async(
                ocr_result['text'], 
                top_k=5
            )
            
            return jsonify({
                'status': 'success',
                'ocr_result': {
                    'text': ocr_result['text'],
                    'confidence': ocr_result['confidence'],
                    'engine_used': ocr_result['engine_used']
                },
                'recommendations': {
                    'products': products,
                    'response': response
                }
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}", exc_info=True)
        return create_error_response('Error processing image', details={
            'ocr_result': {
                'text': '',
                'confidence': 0.0,
                'engine_used': 'easyocr'
            },
            'recommendations': {
                'products': [],
                'response': "Error processing image."
            }
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    services_status = {
        'recommendation_service': 'initialized',
        'ocr_service': 'initialized' if ocr is not None else 'not_initialized',
        'cnn_model': 'initialized' if model is not None else 'not_initialized'
    }
    
    return jsonify({
        'status': 'healthy' if all(status == 'initialized' for status in services_status.values()) else 'degraded',
        'services': services_status
    })

# Initialize model and label mapping
def load_cnn_model():
    try:
        project_root = Path(__file__).parent
        # Load label mapping
        with open(project_root / 'src' / 'data' / 'label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        # Initialize model
        from src.scripts.train_cnn_from_scratch import CNNModel
        num_classes = len(label_mapping)
        model = CNNModel(num_classes)
        
        # Load best model weights
        model_path = project_root / 'models' / 'best_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("CNN model loaded successfully")
        return model, label_mapping, transform
    except Exception as e:
        logger.error(f"Error loading CNN model: {str(e)}")
        return None, None, None

@app.errorhandler(404)
def not_found(error):
    return create_error_response('Resource not found', 404)

@app.errorhandler(500)
def internal_error(error):
    return create_error_response('Internal server error', 500)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Blueprint, request, jsonify, render_template, current_app as app
import asyncio
from functools import wraps
from src.error_handlers import create_error_response

main_bp = Blueprint('main', __name__)

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

@main_bp.route('/')
def home():
    """Home page route."""
    return render_template('index.html')

@main_bp.route('/api/v1/recommendations', methods=['POST'])
@async_route
async def product_recommendation():
    try:
        recommendation_service = app.recommendation_service
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
        app.logger.error(f"Error in product_recommendation endpoint: {str(e)}", exc_info=True)
        return create_error_response('An error occurred while processing your request')

@main_bp.route('/api/v1/ocr-query', methods=['POST'])
@async_route
async def process_ocr_query():
    try:
        ocr = getattr(app, 'ocr', None)
        recommendation_service = app.recommendation_service
        executor = getattr(app, 'executor', None)
        if ocr is None:
            return create_error_response('OCR service not initialized')
        if 'file' not in request.files:
            return create_error_response('No file provided', 400)
        file = request.files['file']
        if file.filename == '':
            return create_error_response('No selected file', 400)
        temp_path = app.config['UPLOAD_FOLDER'] + '/temp.png'
        file.save(temp_path)
        try:
            ocr_result = await asyncio.get_event_loop().run_in_executor(
                executor,
                ocr.process,
                temp_path
            )
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
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        app.logger.error(f"Error in OCR processing: {str(e)}", exc_info=True)
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

@main_bp.route('/api/v1/health', methods=['GET'])
def health_check():
    ocr = getattr(app, 'ocr', None)
    model = getattr(app, 'cnn_model', None)
    services_status = {
        'recommendation_service': 'initialized',
        'ocr_service': 'initialized' if ocr is not None else 'not_initialized',
        'cnn_model': 'initialized' if model is not None else 'not_initialized'
    }
    return jsonify({
        'status': 'healthy' if all(status == 'initialized' for status in services_status.values()) else 'degraded',
        'services': services_status
    }) 
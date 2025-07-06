import os
import asyncio
from flask import Flask
from flask_cors import CORS
from src.utils.logging_utils import init_app_logger
from src.initialization import (
    load_environment,
    create_thread_pool,
    initialize_ocr,
    initialize_services
)
from src.error_handlers import not_found, internal_error
from src.endpoints.routes import main_bp
from src.endpoints.image_detection import image_bp


def create_app() -> Flask:
    # Set up logging using the project utility
    logger = init_app_logger()

    # Use initialization module for environment variables and thread pool
    load_environment()
    executor = create_thread_pool(max_workers=4)

    
    app = Flask(__name__)
    CORS(app)

    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Use initialization module for OCR and services
    ocr = initialize_ocr(use_gpu=False)

    try:
        recommendation_service, vector_db = asyncio.run(initialize_services(executor=executor))
        app.recommendation_service = recommendation_service
    except Exception as e:
        logger.critical(f"Failed to initialize services: {str(e)}")
        raise

    # Register blueprints
    app.register_blueprint(image_bp, url_prefix='/api/v1')
    app.register_blueprint(main_bp, url_prefix='/api/v1')

    # Register error handlers
    app.errorhandler(404)(not_found)
    app.errorhandler(500)(internal_error)

    # Set app attributes for blueprint access
    app.ocr = ocr
    app.executor = executor

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
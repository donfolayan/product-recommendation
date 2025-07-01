import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio
from src.utils.handwriting_ocr import HandwritingOCR
from src.utils.vector_db_utils import VectorDBManager
from src.services.recommendation_service import RecommendationService
from src.utils.model_loader import load_model
from torchvision import transforms
from pathlib import Path

# Configure logging (optionally use setup_logger utility in future refactor)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()

def create_thread_pool(max_workers=4):
    """Create a thread pool executor for async operations."""
    return ThreadPoolExecutor(max_workers=max_workers)

def initialize_ocr(use_gpu=False):
    """Initialize the HandwritingOCR service."""
    try:
        ocr = HandwritingOCR(use_gpu=use_gpu)
        logger.info("OCR initialized successfully")
        return ocr
    except Exception as e:
        logger.error(f"Failed to initialize OCR: {str(e)}")
        return None

async def initialize_services(executor=None):
    """Initialize VectorDBManager and RecommendationService asynchronously."""
    try:
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set!")
        if executor is None:
            executor = create_thread_pool()
        vector_db = VectorDBManager(api_key=pinecone_api_key)
        await asyncio.get_event_loop().run_in_executor(executor, vector_db.initialize)
        recommendation_service = RecommendationService(
            model=vector_db.model,
            index=vector_db.index
        )
        return recommendation_service, vector_db
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}", exc_info=True)
        raise

# Model and label mapping initialization
try:
    model, label_mapping = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None
    label_mapping = None

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 
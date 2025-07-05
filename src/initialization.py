import os
from src.utils.logging_utils import setup_logger
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Optional, Tuple, Dict
from src.utils.handwriting_ocr import HandwritingOCR
from src.utils.vector_db_utils import VectorDBManager
from src.services.recommendation_service import RecommendationService
from src.utils.model_loader import load_model
from src.models.cnn_model import CNNModel
from torchvision import transforms  # type: ignore
from pathlib import Path
from dotenv import load_dotenv
from src.utils.generate_cnn_csv import generate_final_cnn_training_data

# Set up logging using the project utility
logger = setup_logger(__name__, Path('logs/app'))

def load_environment():
    """Load environment variables from .env file using utility."""
    load_dotenv()
    # Ensure the CNN training CSV exists
    csv_path = Path(__file__).parent / 'data' / 'final_cnn_training_data.csv'
    if not csv_path.exists():
        generate_final_cnn_training_data()

def create_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    """Create a thread pool executor for async operations."""
    return ThreadPoolExecutor(max_workers=max_workers)

def initialize_ocr(use_gpu: bool = False) -> Optional[HandwritingOCR]:
    """Initialize the HandwritingOCR service."""
    try:
        ocr = HandwritingOCR(use_gpu=use_gpu)
        logger.info("OCR initialized successfully")
        return ocr
    except Exception as e:
        logger.error(f"Failed to initialize OCR: {str(e)}")
        return None

async def initialize_services(executor: Optional[ThreadPoolExecutor] = None) -> Tuple[RecommendationService, VectorDBManager]:
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

def initialize_model_and_transform() -> Tuple[CNNModel, Dict[str, str], transforms.Compose]:
    """Initialize the CNN model, label mapping, and image transform."""
    try:
        model_path = 'models/best_model.pth'
        label_mapping_path = 'src/data/label_mapping.json'
        model, label_mapping = load_model(model_path, label_mapping_path)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model and transform initialized successfully")
        return model, label_mapping, transform
    except Exception as e:
        logger.error(f"Failed to initialize model and transform: {str(e)}")
        raise

# Initialize model and transform
model, label_mapping, transform = initialize_model_and_transform() 
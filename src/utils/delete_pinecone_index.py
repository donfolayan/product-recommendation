# Import Pinecone and envvar for vector database operations

from pinecone import Pinecone
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_pinecone_index() -> None:
    """Delete the Pinecone index."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set!")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Get index name
        index_name = os.getenv('PINECONE_INDEX_NAME', 'product-vectors')
        
        # Check if index exists
        if index_name in pc.list_indexes().names():
            logger.info(f"Deleting index: {index_name}")
            pc.delete_index(index_name)
            logger.info("Index deleted successfully")
        else:
            logger.info(f"Index {index_name} does not exist")
            
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}")
        raise

if __name__ == "__main__":
    delete_pinecone_index()
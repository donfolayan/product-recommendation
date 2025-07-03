import time
import pandas as pd  # type: ignore
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from typing import List, Optional, Any
from pathlib import Path
from .logging_utils import setup_logger
import sys
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache

# Set up logging
logger = setup_logger(__name__, Path('logs/vector_db'))

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class VectorDBManager:
    """Manages vector database operations for product data."""
    
    api_key: str
    index_name: str
    pc: Pinecone | None
    index: Any
    model: SentenceTransformer | None
    uploaded_ids: list[str]
    progress_dir: Path
    failed_dir: Path
    progress_file: Path
    executor: ThreadPoolExecutor
    vector_cache: TTLCache
    model_cache: TTLCache

    def __init__(self, api_key: str, index_name: str = "product-vectors"):
        """Initialize the VectorDBManager."""
        self.api_key = api_key
        self.index_name = index_name
        self.pc: Optional[Pinecone] = None
        self.index: Optional[Any] = None
        self.model: Optional[SentenceTransformer] = None
        self.uploaded_ids: List[str] = []
        
        # Setup paths relative to project root
        self.progress_dir = project_root / 'src' / 'data' / 'vector_db' / 'progress'
        self.failed_dir = project_root / 'src' / 'data' / 'vector_db' / 'failed'
        self.progress_file = self.progress_dir / 'upload_progress.csv'
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize caches
        self.vector_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
        self.model_cache = TTLCache(maxsize=100, ttl=86400)   # 24 hour cache
        
    def initialize(self) -> None:
        """Initialize Pinecone and the sentence transformer model."""
        try:
            self._initialize_pinecone()
            self._initialize_model()
            self._setup_progress_tracking()
        except Exception:
            logger.error("Failed to initialize VectorDBManager", exc_info=True)
            raise
            
    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone and create index if it doesn't exist."""
        self.pc = Pinecone(api_key=self.api_key)
        
        if not self.pc.has_index(self.index_name):
            logger.info("Creating new Pinecone index")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        else:
            logger.info("Using existing Pinecone index")
            
        self.index = self.pc.Index(self.index_name)
            
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model with caching."""
        if not hasattr(self, 'model') or self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded")
            
    @lru_cache(maxsize=1000)
    def encode_text(self, text: str) -> list[float]:
        """Encode text to vector with caching."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self.model.encode(text).tolist()
            
    def _setup_progress_tracking(self) -> None:
        """Setup directories and load progress tracking."""
        try:
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            self.failed_dir.mkdir(parents=True, exist_ok=True)
            
            if self.progress_file.exists():
                self.uploaded_ids = pd.read_csv(self.progress_file)['StockCode'].tolist()
                logger.info(f"Loaded {len(self.uploaded_ids)} previously uploaded records")
            else:
                self.uploaded_ids = []
                logger.info("Starting fresh upload")
                
        except Exception:
            logger.error("Failed to setup progress tracking", exc_info=True)
            raise
            
    async def create_vector_async(self, row: pd.Series) -> dict[str, Any] | None:
        """Create a vector from a single row of data asynchronously."""
        try:
            # Validate numeric fields
            quantity = float(row['Quantity'])
            unit_price = float(row['UnitPrice'])
            
            # Create product text and generate embedding
            product_text = f"{row['Description']} {row['Country']}"
            
            # Use cached encoding if available
            cache_key = f"vector_{hash(product_text)}"
            if cache_key in self.vector_cache:
                embedding = self.vector_cache[cache_key]
            else:
                # Run encoding in thread pool
                if self.model is None:
                    raise RuntimeError("Model not initialized. Call initialize() first.")
                embedding = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.model.encode,
                    product_text
                )
                self.vector_cache[cache_key] = embedding
            
            return {
                "id": str(row['StockCode']),
                "values": embedding.tolist(),
                "metadata": {
                    "InvoiceNo": str(row['InvoiceNo']),
                    "CustomerID": str(row['CustomerID']),
                    "Quantity": quantity,
                    "UnitPrice": unit_price,
                    "Description": str(row['Description']),
                    "Country": str(row['Country'])
                }
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid numeric value in row {row['StockCode']}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error creating vector for {row['StockCode']}: {str(e)}")
            return None
            
    async def process_chunk_async(self, chunk_df: pd.DataFrame) -> None:
        """Process a chunk of data and upload to Pinecone asynchronously."""
        try:
            # Filter out already uploaded records
            chunk_df = chunk_df[~chunk_df['StockCode'].isin(self.uploaded_ids)]
            
            if len(chunk_df) == 0:
                return
                
            # Create vectors asynchronously
            vectors = []
            for _, row in chunk_df.iterrows():
                vector = await self.create_vector_async(row)
                if vector:
                    vectors.append(vector)
            
            if vectors:
                await self._upload_vectors_async(vectors)
                
        except Exception:
            logger.error("Error processing chunk", exc_info=True)
                
    async def _upload_vectors_async(self, vectors: list[dict[str, Any]]) -> None:
        """Upload vectors to Pinecone and update progress tracking asynchronously."""
        try:
            # Run Pinecone upsert in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.upsert(vectors=vectors)
            )
            logger.info(f"Uploaded {len(vectors)} vectors")
            
            # Update progress tracking
            successful_ids = [v['id'] for v in vectors]
            self.uploaded_ids.extend(successful_ids)
            
            # Run CSV update in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: pd.DataFrame({'StockCode': self.uploaded_ids}).to_csv(self.progress_file, index=False)
            )
            
        except Exception:
            logger.error("Error uploading vectors", exc_info=True)
            await self._save_failed_records_async(vectors)
                
    async def _save_failed_records_async(self, vectors: list[dict[str, Any]]) -> None:
        """Save failed records for retry asynchronously."""
        try:
            failed_records = pd.DataFrame([{
                'StockCode': v['id'],
                'InvoiceNo': v['metadata']['InvoiceNo'],
                'CustomerID': v['metadata']['CustomerID'],
                'Quantity': v['metadata']['Quantity'],
                'UnitPrice': v['metadata']['UnitPrice'],
                'Description': v['metadata']['Description'],
                'Country': v['metadata']['Country']
            } for v in vectors])
            
            timestamp = int(time.time())
            file_path = self.failed_dir / f'failed_{timestamp}.csv'
            
            # Run CSV save in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: failed_records.to_csv(file_path, index=False)
            )
            logger.warning(f"Saved {len(vectors)} failed records")
            
        except Exception:
            logger.error("Error saving failed records", exc_info=True)
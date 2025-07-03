import re
from typing import Any
from src.utils.logging_utils import setup_logger
from sentence_transformers import SentenceTransformer
import os
import google.generativeai as genai
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path

# Set up logging using the project utility
logger = setup_logger(__name__, Path('logs/recommendation_service'))

class RecommendationService:
    model: SentenceTransformer
    index: Any
    sensitive_patterns: list[str]
    max_query_length: int
    min_query_length: int
    executor: ThreadPoolExecutor
    cache: TTLCache
    gemini_api_key: str | None
    gemini_client: genai.GenerativeModel | None

    def __init__(self, model: SentenceTransformer, index: Any):
        self.model = model
        self.index = index
        self.sensitive_patterns = [
            r'\b(credit|card|password|ssn|social|security)\b',
            r'\b\d{16}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        ]
        self.max_query_length = 500
        self.min_query_length = 3
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize cache with longer TTL for better performance
        self.cache = TTLCache(maxsize=2000, ttl=7200)  # 2 hour cache
        
        # Initialize Gemini with rate limiting
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini API configured for recommendation service")
        else:
            logger.warning("GEMINI_API_KEY not found; natural language responses will be disabled")
            self.gemini_client = None

    @lru_cache(maxsize=1000)
    def _validate_query(self, query: str) -> tuple[bool, str]:
        """Validate the query for security and quality with caching."""
        if not query or not isinstance(query, str):
            return False, "Please provide a valid search query."

        query = query.strip()
        
        # Check query length
        if len(query) < self.min_query_length:
            return False, f"Query must be at least {self.min_query_length} characters long."
        if len(query) > self.max_query_length:
            return False, f"Query must not exceed {self.max_query_length} characters."

        # Check for sensitive information
        for pattern in self.sensitive_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains sensitive information that cannot be processed."

        # Check for SQL injection attempts
        sql_patterns = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
        if any(pattern in query.lower() for pattern in sql_patterns):
            return False, "Invalid query format detected."

        return True, query

    def _sanitize_product_data(self, product: dict) -> dict:
        """Sanitize product data."""
        try:
            return {
                'id': str(product.get('id', '')),
                'description': str(product.get('description', '')),
                'price': float(product.get('price', 0.0)),
                'country': str(product.get('country', '')),
                'score': float(product.get('score', 0.0))
            }
        except Exception as e:
            logger.error(f"Error sanitizing product data: {str(e)}")
            return {
                'id': str(product.get('id', '')),
                'description': str(product.get('description', '')),
                'price': 0.0,
                'country': str(product.get('country', '')),
                'score': 0.0
            }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_response(self, products: list[dict[str, Any]], query: str) -> str:
        """Generate response using Gemini or fallback to basic description."""
        if not products:
            return "No matching products found."
            
        if not self.gemini_client:
            return self._generate_fallback_response(products)
            
        try:
            # Format products for the prompt
            products_text = "\n".join([
                f"{i+1}. {p['description']} - ${p['price']:.2f}"
                for i, p in enumerate(products)
            ])
            
            prompt = f"""Based on the query "{query}", here are the recommended products:
            {products_text}

            Please provide a natural, flowing description of each product. Follow these guidelines:
            1. Do not add an introduction, just answer straightaway
            2. Write each product description as a complete sentence
            3. Focus on the key benefits and features that make each product special
            4. Use natural, engaging language
            5. Keep each description concise but informative
            6. Do not use bullet points, numbers, or any formatting
            7. Do not mention prices
            8. Write in a way that flows naturally from one product to the next
            9. Use plain text only, no HTML or markdown"""

            # Run Gemini API call in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.gemini_client.generate_content,
                prompt
            )
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
            
        except Exception as e:
            logger.error(f"Gemini response generation failed: {str(e)}")
            return self._generate_fallback_response(products)

    def _generate_fallback_response(self, products: list[dict]) -> str:
        """Generate a fallback response when Gemini is not available."""
        if not products:
            return "No matching products found."

        descriptions = []
        for product in products:
            description = product.get('description', '')
            if description:
                descriptions.append(description)

        return " ".join(descriptions)

    async def get_recommendations_async(self, query: str, top_k: int = 5) -> tuple[list[dict], str]:
        """
        Get product recommendations asynchronously with caching and rate limiting.
        
        Args:
            query (str): Natural language query
            top_k (int): Number of recommendations to return
            
        Returns:
            Tuple[List[Dict], str]: (products, response)
        """
        try:
            # Check cache first
            cache_key = f"{query}_{top_k}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Validate query
            is_valid, message = self._validate_query(query)
            if not is_valid:
                return [], message

            # Generate query vector
            query_vector = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.model.encode,
                query
            )

            # Query Pinecone
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.query(
                    vector=query_vector.tolist(),
                    top_k=top_k,
                    include_metadata=True
                )
            )

            # Process and sanitize results
            products = []
            for match in results.matches:
                product = {
                    'id': match.id,
                    'score': match.score,
                    'description': match.metadata.get('Description', 'No description'),
                    'price': float(match.metadata.get('UnitPrice', 0)),
                    'country': match.metadata.get('Country', 'Unknown')
                }
                products.append(self._sanitize_product_data(product))

            # Generate response
            response = await self._generate_response(products, query)
            
            # Cache the results
            self.cache[cache_key] = (products, response)
            
            return products, response

        except Exception as e:
            logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
            return [], "An error occurred while processing your request. Please try again later."

    def get_recommendations(self, query: str, top_k: int = 5) -> tuple[list[dict], str]:
        """
        Synchronous wrapper for get_recommendations_async.
        """
        return asyncio.run(self.get_recommendations_async(query, top_k))
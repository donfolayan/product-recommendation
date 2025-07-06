import re
import torch
import logging
from typing import Optional
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSpellChecker:
    spell_checker: SpellChecker
    model: Optional[SentenceTransformer]
    product_dictionary: set[str]
    ocr_patterns: dict[str, str]

    def __init__(self, model: Optional[SentenceTransformer] = None):
        """Initialize the enhanced spell checker.
        
        Args:
            model: Optional sentence transformer model for context-aware corrections
        """
        self.spell_checker = SpellChecker()
        self.model = model
        self.product_dictionary = set()
        self.ocr_patterns = defaultdict(str)
        self._initialize_patterns()
        self._load_product_terms()
        
        # Initialize BERT if not provided
        if self.model is None:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
                logger.info("Initialized BERT model for context understanding")
            except Exception as e:
                logger.warning(f"Could not initialize BERT model: {str(e)}")
                self.model = None
        
    def _initialize_patterns(self) -> None:
        """Initialize common OCR patterns and corrections."""
        # Common OCR/typing patterns
        self.ocr_patterns.update({
            r'\bT\b(?!-)': 'I',  # Only convert T to I if not followed by hyphen
            r'W:NT': 'want',
            r'WNT': 'want',
            r'WANT': 'want',
            # Product-specific patterns
            r'T-SHL': 'T-Shirt',
            r'TSHL': 'T-Shirt',
            r'T-SHIRT': 'T-Shirt',
            r'TSHIRT': 'T-Shirt',
            r'TSHIRTS': 'T-Shirts',
            r'T-SHIRTS': 'T-Shirts',
            r'I-SHL': 'T-Shirt',  # Handle case where T was converted to I
            r'ISHIRT': 'T-Shirt',
            r'I-SHIRT': 'T-Shirt',
            # Common product categories
            r'HDPHN': 'headphone',
            r'HDPHNS': 'headphones',
            r'MOUS': 'mouse',
            r'ESP': 'espresso',
            r'FIT': 'fitness',
            r'TRACK': 'tracker',
        })
        
    def _load_product_terms(self) -> None:
        """Load product terms from the dataset."""
        try:
            terms_path = Path(__file__).parent.parent / 'data' / 'dataset' / 'product_terms.txt'
            if terms_path.exists():
                with open(terms_path, 'r', encoding='utf-8') as f:
                    terms = {line.strip().lower() for line in f if line.strip()}
                self.add_product_terms(list(terms))
                logger.info(f"Loaded {len(terms)} product terms from {terms_path}")
            else:
                logger.warning(f"Product terms file not found at {terms_path}")
        except Exception as e:
            logger.error(f"Error loading product terms: {str(e)}")
            
    def add_product_terms(self, terms: list[str]) -> None:
        """Add product-specific terms to the dictionary.
        
        Args:
            terms: List of product terms to add
        """
        for term in terms:
            if not term or not isinstance(term, str):
                continue
            # Add the term and its variations
            self.product_dictionary.add(term.lower())
            self.product_dictionary.add(term.lower().replace('-', ' '))
            self.product_dictionary.add(term.lower().replace(' ', '-'))
            # Add individual words for partial matching
            self.product_dictionary.update(word.lower() for word in term.split())
            
    def _apply_pattern_corrections(self, text: str) -> str:
        """Apply pattern-based corrections.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text
        """
        corrected_text = text
        for pattern, replacement in self.ocr_patterns.items():
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
            if corrected_text != text:
                logger.debug(f"Pattern correction: '{text}' -> '{corrected_text}'")
                text = corrected_text
        return corrected_text
        
    def _get_context_similarity(self, word: str, context: str) -> float:
        """Get similarity score between word and context using BERT.
        
        Args:
            word: Word to check
            context: Surrounding context
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.model:
            return 0.0
            
        try:
            # Clean and normalize the context
            context = self._clean_context(context)
            
            # If context is too noisy, return low confidence
            if self._is_noisy_context(context):
                return 0.3  # Low but not zero confidence
                
            # Get embeddings
            with torch.no_grad():  # Disable gradient calculation
                word_embedding = self.model.encode(word, convert_to_tensor=True)
                context_embedding = self.model.encode(context, convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    word_embedding.unsqueeze(0),
                    context_embedding.unsqueeze(0)
                ).item()
                
                # Adjust confidence based on context quality
                if self._is_product_context(context):
                    similarity *= 1.2  # Boost confidence for product-related context
                    
                return min(1.0, max(0.0, similarity))
                
        except Exception as e:
            logger.error(f"Error calculating context similarity: {str(e)}")
            return 0.0
            
    def _clean_context(self, context: str) -> str:
        """Clean and normalize context for BERT processing.
        
        Args:
            context: Raw context string
            
        Returns:
            Cleaned context string
        """
        # Remove special characters but keep spaces and hyphens
        context = re.sub(r'[^a-zA-Z0-9\s-]', ' ', context)
        # Normalize whitespace
        context = ' '.join(context.split())
        return context
        
    def _is_noisy_context(self, context: str) -> bool:
        """Check if context is too noisy for reliable BERT processing.
        
        Args:
            context: Context string to check
            
        Returns:
            True if context is too noisy
        """
        # Check for very short context
        if len(context.split()) < 2:
            return True
            
        # Check for too many numbers or special characters
        if sum(c.isdigit() for c in context) / len(context) > 0.3:
            return True
            
        # Check for repeated characters (common in OCR errors)
        if any(len(set(word)) < len(word) * 0.5 for word in context.split()):
            return True
            
        return False
        
    def _is_product_context(self, context: str) -> bool:
        """Check if context is product-related.
        
        Args:
            context: Context string to check
            
        Returns:
            True if context is product-related
        """
        # Check if any product terms are in the context
        context_words = set(context.lower().split())
        return any(term in context_words for term in self.product_dictionary)
        
    def correct(self, text: str) -> str:
        """Correct text using hybrid approach.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text
        """
        try:
            # First apply pattern-based corrections
            text = self._apply_pattern_corrections(text)
            logger.debug(f"After pattern corrections: {text}")
            
            # Split into words
            words = text.split()
            corrected_words = []
            
            for i, word in enumerate(words):
                # Skip if word is in product dictionary
                if word.lower() in self.product_dictionary:
                    corrected_words.append(word)
                    continue
                    
                # Get context (surrounding words)
                context = ' '.join(words[max(0, i-2):min(len(words), i+3)])
                
                # If word is misspelled
                if word.isalpha() and word not in self.spell_checker:
                    # Get possible corrections
                    candidates = self.spell_checker.candidates(word)
                    if not candidates:
                        corrected_words.append(word)
                        continue
                        
                    # Score each candidate
                    best_candidate = word
                    best_score = 0.0
                    
                    for candidate in candidates:
                        # Calculate score based on:
                        # 1. Context similarity (if model available)
                        # 2. Whether it's in product dictionary
                        # 3. Original spell checker confidence
                        context_score = self._get_context_similarity(candidate, context)
                        dict_score = 1.0 if candidate.lower() in self.product_dictionary else 0.0
                        
                        # Combined score (can be adjusted)
                        # Give more weight to dictionary matches for product terms
                        if dict_score > 0:
                            score = (0.3 * context_score + 0.7 * dict_score)
                        else:
                            score = (0.6 * context_score + 0.4 * dict_score)
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = candidate
                    
                    # Only apply correction if confidence is high enough
                    # Higher threshold for non-product terms
                    threshold = 0.7 if best_candidate.lower() in self.product_dictionary else 0.8
                    if best_score > threshold:
                        corrected_words.append(best_candidate)
                        logger.debug(f"Corrected '{word}' to '{best_candidate}' (score: {best_score:.2f})")
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
            
            corrected_text = ' '.join(corrected_words)
            logger.debug(f"Final corrected text: {corrected_text}")
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error in text correction: {str(e)}")
            return text 
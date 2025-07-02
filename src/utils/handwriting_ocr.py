import os
import logging
import torch
import easyocr  # type: ignore
import numpy as np
from PIL import Image  # type: ignore
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, Any, cast
import cv2
from .enhanced_spell_checker import EnhancedSpellChecker
import re
from collections import defaultdict
import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv
from .logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__, Path('logs'))

class HandwritingOCR:
    device: torch.device
    reader: easyocr.Reader
    spell_checker: EnhancedSpellChecker
    use_gemini: bool
    gemini_api_key: str | None
    gemini_client: genai.GenerativeModel | None

    def __init__(self, use_gpu: bool = False, use_gemini: bool = True):
        """Initialize the HandwritingOCR class.
        
        Args:
            use_gpu (bool): Whether to use GPU if available (default: False for CPU)
            use_gemini (bool): Whether to use Gemini API for text cleanup (default: True)
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        logger.info("EasyOCR initialized")
        
        # Initialize enhanced spell checker
        self.spell_checker = EnhancedSpellChecker()
        logger.info("Spell checker initialized")

        self.use_gemini = use_gemini
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.use_gemini and not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found; Gemini cleanup disabled")
            self.use_gemini = False
        if self.use_gemini:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini API configured")

    def add_product_terms(self, terms: list[str]) -> None:
        """Add product-specific terms to the spell checker."""
        self.spell_checker.add_product_terms(terms)
        logger.info(f"Added {len(terms)} product terms")

    def preprocess_image(self, image: Any) -> Any:
        """Preprocess image for better OCR results."""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, str):
                pil_image = cast(Image.Image, Image.open(image))
            elif isinstance(image, np.ndarray):
                pil_image = cast(Image.Image, Image.fromarray(image))
            else:
                pil_image = cast(Image.Image, image)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array for OpenCV processing
            cv_image = np.array(pil_image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Invert for line detection
            inverted_thresh = cv2.bitwise_not(thresh)

            # Detect horizontal lines
            lines = cv2.HoughLinesP(inverted_thresh, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
            
            # Create copy for line removal
            image_without_lines = inverted_thresh.copy()
            
            # Remove detected lines
            if lines is not None:
                for line in lines:
                    if isinstance(line, (list, np.ndarray)) and len(line) > 0:
                        x1, y1, x2, y2 = line[0]
                        if abs(y2 - y1) < 5:
                            cv2.line(image_without_lines, (x1, y1), (x2, y2), (0, 0, 0), 5)
            
            # Invert back
            final_processed_image = cv2.bitwise_not(image_without_lines)
            
            # Dilate text
            kernel = np.ones((2,2),np.uint8)
            dilated = cv2.dilate(final_processed_image, kernel, iterations = 1)
            
            # Convert back to RGB
            rgb_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
            
            # Save debug image
            debug_path = Path('debug')
            debug_path.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_path / 'preprocessed.png'), rgb_image)
            
            return rgb_image
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise

    def run_easyocr(self, image: np.ndarray) -> list[tuple[list[list[int]], str, float]]:
        """Run EasyOCR on the image."""
        try:
            results = self.reader.readtext(image, detail=1, paragraph=False)
            return results
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return []

    def _get_pattern_corrections(self) -> dict[str, str]:
        """Get all pattern corrections including OCR, typing, and product-specific patterns.
        
        Returns:
            Dictionary of patterns and their corrections
        """
        return {
            # Product-specific patterns
            r'T-SHL': 'T-Shirt',
            r'TSHL': 'T-Shirt',
            r'T-SHIRT': 'T-Shirt',
            r'TSHIRT': 'T-Shirt',
            r'TSHIRTS': 'T-Shirts',
            r'T-SHIRTS': 'T-Shirts',
            r'I-SHL': 'T-Shirt',
            r'ISHIRT': 'T-Shirt',
            r'I-SHIRT': 'T-Shirt',
            r'HDPHN': 'headphone',
            r'HDPHNS': 'headphones',
            r'MOUS': 'mouse',
            r'ESP': 'espresso',
            r'FIT': 'fitness',
            r'TRACK': 'tracker',
            
            # Common OCR/typing substitutions
            r'\bT\b(?!-)': 'I',
            r'W:NT': 'want',
            r'WNT': 'want',
            r'WANT': 'want',
            r'C\s*ho\s*c': 'Choc',
            r'0': 'o',
            r'lat': 'late',
            r'&': 'and',
            r'%': 'e',
            r'Ca\s*k': 'Cake',
            
            # Number to word conversions
            r'1': 'one',
            r'2': 'two',
            r'3': 'three',
            r'4': 'four',
            r'5': 'five',
        }

    def _get_product_combinations(self) -> dict[str, list[str]]:
        """Get common product combinations and their variations."""
        return {
            'chocolate cake': ['chocolate & cake', 'choc cake', 'chocolate cake', 'choc & cake'],
            'coffee maker': ['coffee & maker', 'coffee maker', 'coffee machine', 'coffee & machine'],
            'wireless mouse': ['wireless & mouse', 'wireless mouse'],
        }

    def _apply_corrections(self, text: str) -> str:
        """Apply all text corrections including patterns and spell checking."""
        try:
            # Apply pattern corrections
            patterns = self._get_pattern_corrections()
            for pattern, replacement in patterns.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            # Apply spell checking
            text = self.spell_checker.correct(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Text correction failed: {str(e)}")
            return text

    def gemini_cleanup(self, text: str, image: Any = None) -> str:
        """Use Gemini API to clean up noisy OCR text."""
        if not self.use_gemini or not self.gemini_api_key or not text.strip():
            return text
            
        try:
            prompt = f"The following is noisy OCR output from handwriting. Please correct it to the most likely intended English phrase.\nOCR: {text}\nCorrected:"
            contents_parts: list[Union[str, dict[str, Any]]] = [prompt]
            
            if image is not None:
                if isinstance(image, str):
                    img = cast(Image.Image, Image.open(image))
                elif isinstance(image, np.ndarray):
                    img = cast(Image.Image, Image.fromarray(image))
                else:
                    img = cast(Image.Image, image)
                    
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format='JPEG')
                contents_parts.insert(0, {"mime_type": "image/jpeg", "data": buf.getvalue()})

            if self.gemini_client is None:
                raise RuntimeError("Gemini client not initialized")
            response = self.gemini_client.generate_content(contents_parts)
            cleaned = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            logger.info(f"Gemini cleaned text: {cleaned}")
            return cleaned
            
        except Exception as e:
            logger.error(f"Gemini cleanup failed: {str(e)}")
            return text

    def process(self, image: Any, easyocr_confidence_threshold: float = 0.1) -> dict[str, Any]:
        """Process an image to detect handwritten text using EasyOCR.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            easyocr_confidence_threshold: Minimum confidence threshold for EasyOCR
            
        Returns:
            Dictionary containing:
            - text: Detected text
            - confidence: Average confidence score
            - engine_used: Which OCR engine was used
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run EasyOCR
            logger.debug("Running EasyOCR on processed image")
            results = self.run_easyocr(processed_image)
            
            if not results:
                logger.warning("No text detected")
                return self._create_empty_result()
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            for result in results:
                if len(result) == 3:
                    texts.append(result[1])
                    confidences.append(result[2])
                else:
                    texts.append(result[1])
                    confidences.append(0.0)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            logger.debug(f"Average confidence: {avg_confidence}")
            
            # Process text
            text = " ".join(texts)
            text = text.replace('|', 'I').replace('[', 'C').strip()
            
            # Apply corrections
            text = self._apply_corrections(text)
            
            # Use Gemini for cleanup if enabled
            if self.use_gemini and text.strip():
                text = self.gemini_cleanup(text, image)
            
            if avg_confidence >= easyocr_confidence_threshold:
                return {
                    'text': text,
                    'confidence': avg_confidence,
                    'engine_used': 'easyocr'
                }
            
            logger.warning("Confidence too low")
            return self._create_empty_result()
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> dict[str, str | float]:
        """Create an empty result dictionary."""
        return {
            'text': '',
            'confidence': 0.0,
            'engine_used': 'easyocr'
        } 
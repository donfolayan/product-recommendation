import os
from src.utils.logging_utils import setup_logger
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import numpy as np
from skimage.metrics import structural_similarity as ssim
import traceback
from urllib.parse import quote_plus
import pandas as pd
import random
import time
import json
from datetime import datetime
import aiohttp
import asyncio
import io
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Ensure log directory exists
log_dir = Path(__file__).resolve().parent.parent.parent / 'logs/scraper'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(__name__, log_dir)
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]
ACCEPT_LANGUAGES = [
    'en-US,en;q=0.9',
    'en-GB,en;q=0.8',
    'en;q=0.7',
    'en-US;q=0.6',
    'en-CA,en;q=0.5'
]

class ImageSimilarityChecker:
    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def is_unique(self, img: Image.Image, existing_images: List[Image.Image]) -> bool:
        try:
            img1 = img.convert('L').resize((100, 100))
            arr1 = np.array(img1)
            for existing_img in existing_images:
                img2 = existing_img.convert('L').resize((100, 100))
                arr2 = np.array(img2)
                similarity = ssim(arr1, arr2)
                if similarity > self.threshold:
                    return False
            return True
        except Exception as e:
            logger.error(f"Error in is_unique: {str(e)}")
            return False

class AmazonScraper:
    def __init__(self) -> None:
        self.base_url = "https://www.amazon.com"
        self.search_url = f"{self.base_url}/s"
        self.session = requests.Session()
        self.rate_limit = {
            'last_request': datetime.now(),
            'min_delay': 5.0,
            'max_delay': 15.0,
            'backoff_factor': 1.5,
            'current_delay': 5.0,
            'consecutive_failures': 0
        }
        self.max_workers = 5
        self.similarity_checker = ImageSimilarityChecker()

    def _respect_rate_limit(self) -> None:
        """Implement rate limiting with random delay between min and max delay"""
        now = datetime.now()
        time_since_last = (now - self.rate_limit['last_request']).total_seconds()
        if time_since_last < self.rate_limit['current_delay']:
            sleep_time = self.rate_limit['current_delay'] - time_since_last
            time.sleep(sleep_time)
        self.rate_limit['last_request'] = datetime.now()
        self.rate_limit['current_delay'] = random.uniform(self.rate_limit['min_delay'], self.rate_limit['max_delay'])

    def _handle_rate_limit(self, success: bool) -> None:
        """Update rate limiting parameters based on request success"""
        if success:
            self.rate_limit['current_delay'] = random.uniform(self.rate_limit['min_delay'], self.rate_limit['max_delay'])
            self.rate_limit['consecutive_failures'] = 0
        else:
            self.rate_limit['consecutive_failures'] += 1
            self.rate_limit['current_delay'] = min(
                self.rate_limit['max_delay'],
                self.rate_limit['current_delay'] * self.rate_limit['backoff_factor']
            )
            if self.rate_limit['consecutive_failures'] > 3:
                extra_delay = random.uniform(5, 10)
                time.sleep(extra_delay)

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with rate limiting and error handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self._respect_rate_limit()
                headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': random.choice(ACCEPT_LANGUAGES),
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                response = self.session.get(url, headers=headers, timeout=60)
                
                if "Enter the characters you see below" in response.text or "Type the characters you see in this image" in response.text:
                    logger.warning("Bot detection triggered, increasing delay")
                    self._handle_rate_limit(False)
                    retry_count += 1
                    continue
                
                if response.status_code == 200:
                    self._handle_rate_limit(True)
                    return response
                elif response.status_code == 503:
                    logger.warning("Service unavailable, backing off")
                    self._handle_rate_limit(False)
                    retry_count += 1
                    continue
                else:
                    logger.error(f"Request failed with status code: {response.status_code}")
                    self._handle_rate_limit(False)
                    retry_count += 1
                    
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                self._handle_rate_limit(False)
                retry_count += 1
                
        return None

    async def _download_image_async(self, session: aiohttp.ClientSession, url: str, save_path: str) -> bool:
        """Download image asynchronously"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    img = Image.open(io.BytesIO(content))
                    img.save(save_path, 'JPEG', quality=85, optimize=True)
                    return True
        except Exception as e:
            logger.error(f"Failed to download image {url}: {str(e)}")
        return False

    async def _process_images_async(self, image_urls: List[str], image_dir: str, existing_thumbnails: List[Image.Image], metadata: List[Dict[str, Any]]) -> None:
        """Process multiple images concurrently"""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = []
            for url in image_urls:
                if len(metadata) >= 50:
                    break
                    
                filename = f"amazon_{len(metadata) + 1}.jpg"
                save_path = os.path.join(image_dir, filename)
                
                if os.path.exists(save_path):
                    continue
                
                task = asyncio.create_task(self._download_image_async(session, url, save_path))
                tasks.append((task, url, save_path))
            
            for task, url, save_path in tasks:
                try:
                    success = await task
                    if success:
                        img = Image.open(save_path)
                        if self.similarity_checker.is_unique(img, existing_thumbnails):
                            metadata.append({
                                'filename': os.path.basename(save_path),
                                'url': url,
                                'timestamp': datetime.now().isoformat()
                            })
                            existing_thumbnails.append(img)
                            
                            if len(metadata) % 5 == 0:
                                self._save_metadata(image_dir, metadata)
                                
                except Exception as e:
                    logger.error(f"Error processing image {url}: {str(e)}")
                    if os.path.exists(save_path):
                        os.remove(save_path)

    def _generate_search_terms(self, description: str) -> List[str]:
        """Generate alternative search terms from the description, focusing on natural product search patterns"""
        terms = [description]
        
        desc_lower = description.lower()
        words = desc_lower.split()
        
        if len(words) >= 2:
            colors = ['pink', 'red', 'blue', 'green', 'black', 'white', 'chocolate', 'rustic']
            patterns = ['polkadot', 'spotty', 'woodland', 'retrospot']
            
            color_words = [w for w in words if w in colors]
            other_words = [w for w in words if w not in color_words]
            
            if 'bag' in desc_lower:
                if 'lunch' in desc_lower:
                    terms.append(f"lunch bag {desc_lower.replace('lunch bag', '').strip()}")
                elif 'storage' in desc_lower:
                    terms.append(f"storage bag {desc_lower.replace('storage bag', '').strip()}")
                elif 'shopper' in desc_lower:
                    terms.append(f"shopping bag {desc_lower.replace('shopper', '').strip()}")
            
            if 'clock' in desc_lower:
                terms.append(f"alarm clock {desc_lower.replace('clock', '').strip()}")
            
            if 'bottle' in desc_lower:
                terms.append(f"water bottle {desc_lower.replace('bottle', '').strip()}")
            
            if 'bunting' in desc_lower:
                terms.append(f"bunting {desc_lower.replace('bunting', '').strip()}")
            
            if 'tea set' in desc_lower:
                terms.append(f"tea set {desc_lower.replace('tea set', '').strip()}")
            
            if 'ribbon' in desc_lower:
                terms.append(f"ribbon {desc_lower.replace('ribbon', '').strip()}")
            
            if 'cake' in desc_lower:
                terms.append(f"cake stand {desc_lower.replace('cake', '').strip()}")
            
            if color_words and other_words:
                terms.append(' '.join(other_words + color_words))
            
            pattern_words = [w for w in words if w in patterns]
            if pattern_words and other_words:
                terms.append(' '.join(other_words + pattern_words))
        
        terms = list(set(term.strip() for term in terms if term.strip()))
        logger.info(f"Generated search terms: {terms}")
        return terms

    def _parse_srcset(self, srcset: str) -> List[Tuple[str, int, int]]:
        images = []
        for src in srcset.split(','):
            try:
                url, size = src.strip().split(' ')
                multiplier = float(size.replace('x', ''))
                base_width = 320
                width = int(base_width * multiplier)
                height = int(width * 0.75)
                images.append((url, width, height))
            except (ValueError, IndexError):
                continue
        return sorted(images, key=lambda x: x[1], reverse=True)

    def _download_thumbnail(self, url: str) -> Optional[Image.Image]:
        try:
            response = self._make_request(url)
            if not response:
                return None
            img = Image.open(BytesIO(response.content))
            img.thumbnail((100, 100))
            return img
        except Exception as e:
            logger.error(f"Error downloading thumbnail: {str(e)}")
            return None

    def search_and_download_images(self, description: str, stock_code: str) -> bool:
        logger.info(f"[START] search_and_download_images for stock_code={stock_code}, description='{description}'")
        try:
            base_dir = os.path.join(project_root, "static", "images")
            image_dir = os.path.join(base_dir, str(stock_code))
            try:
                if not os.path.exists(base_dir):
                    logger.info(f"Creating base directory: {base_dir}")
                    os.makedirs(base_dir)
                if not os.path.exists(image_dir):
                    logger.info(f"Creating stock code directory: {image_dir}")
                    os.makedirs(image_dir)
                logger.info(f"[INFO] Files will be saved to: {image_dir}")
            except Exception as e:
                logger.error(f"Failed to create directories: {str(e)}")
                return False

            metadata_file = os.path.join(image_dir, "metadata.json")
            existing_metadata = []
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        existing_metadata = json.load(f)
                    logger.info(f"Loaded existing metadata for {stock_code} with {len(existing_metadata)} entries")
                except Exception as e:
                    logger.error(f"Error loading metadata: {str(e)}")

            existing_files = [f for f in os.listdir(image_dir) if f.startswith('amazon_') and f.endswith('.jpg')]
            saved_count = len(existing_files)
            logger.info(f"Found {saved_count} existing amazon_*.jpg files")

            if saved_count >= 50:
                logger.info("Already have 50 images, skipping search")
                return True

            search_terms = self._generate_search_terms(description)
            downloaded_images = []
            existing_thumbnails = []
            
            for term in search_terms:
                if saved_count >= 50:
                    logger.info("Already have 50 images, stopping search")
                    break
                    
                logger.info(f"Searching for term: {term}")
                encoded_term = quote_plus(term)
                url = f"{self.search_url}?k={encoded_term}"
                response = self._make_request(url)
                if not response:
                    logger.warning(f"No response for term: {term}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                img_elements = soup.select('img.s-image')
                logger.info(f"Found {len(img_elements)} images for term: {term}")
                
                for img in img_elements:
                    if saved_count >= 50:
                        logger.info("Already have 50 images, stopping image processing")
                        break
                        
                    try:
                        srcset = img.get('srcset', '')
                        if not srcset:
                            continue
                        image_sizes = self._parse_srcset(srcset)
                        if not image_sizes:
                            continue
                        thumbnail_url = image_sizes[-1][0]
                        thumbnail = self._download_thumbnail(thumbnail_url)
                        if not thumbnail:
                            continue
                        if not self.similarity_checker.is_unique(thumbnail, existing_thumbnails):
                            logger.info(f"Skipping duplicate image: {thumbnail_url}")
                            continue
                        high_quality_url = image_sizes[0][0]
                        response = self._make_request(high_quality_url)
                        if not response:
                            continue
                        image_data = response.content
                        width, height = image_sizes[0][1], image_sizes[0][2]
                        
                        image_path = os.path.join(image_dir, f"amazon_{saved_count + 1}.jpg")
                        if not os.path.exists(image_path):
                            try:
                                with open(image_path, 'wb') as f:
                                    f.write(image_data)
                                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                                    logger.info(f"[SUCCESS] Saved image {saved_count + 1} at {image_path} ({width}x{height})")
                                    
                                    image_metadata = {
                                        "image_number": saved_count + 1,
                                        "filename": f"amazon_{saved_count + 1}.jpg",
                                        "url": high_quality_url,
                                        "thumbnail_url": thumbnail_url,
                                        "width": width,
                                        "height": height,
                                        "search_term": term,
                                        "downloaded_at": datetime.now().isoformat()
                                    }
                                    existing_metadata.append(image_metadata)
                                    
                                    with open(metadata_file, 'w') as f:
                                        json.dump(existing_metadata, f, indent=2)
                                    
                                    saved_count += 1
                                    downloaded_images.append((image_data, high_quality_url, width, height))
                                    existing_thumbnails.append(thumbnail)
                                else:
                                    logger.error(f"[FAIL] File not saved or empty: {image_path}")
                            except Exception as e:
                                logger.error(f"[ERROR] Failed to save image to {image_path}: {str(e)}")
                                continue
                        else:
                            logger.info(f"Image already exists, skipping: {image_path}")
                            
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
                        continue

            logger.info(f"[SUMMARY] stock_code={stock_code}: {saved_count} images saved out of {len(downloaded_images)} attempted.")
            logger.info(f"[END] search_and_download_images for stock_code={stock_code}")
            return saved_count > 0
        except Exception as e:
            logger.error(f"[FATAL ERROR] in search_and_download_images: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def scrape_images_from_file(input_file: str) -> None:
    """Scrape images for all products in the input file"""
    try:
        df = pd.read_csv(input_file)
        
        scraper = AmazonScraper()
        
        for index, row in df.iterrows():
            stock_code = str(row['StockCode'])
            description = row['Description']
            logger.info(f"Processing stock code: {stock_code}")
            success = scraper.search_and_download_images(description, stock_code)
            if not success:
                logger.error(f"Failed to process {description} (stock code: {stock_code})")
            time.sleep(random.uniform(2, 5))
    except Exception as e:
        logger.error(f"Error in scrape_images_from_file: {str(e)}")
        logger.error(traceback.format_exc())

def main() -> None:
    import sys
    if len(sys.argv) != 2:
        logger.error("Usage: python web_scraping_fix.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    scrape_images_from_file(input_file)

if __name__ == "__main__":
    main() 
# Product Recommendation, OCR, and Image-Based Detection System

## Overview
This project is a modular system for product recommendation, OCR-based query processing, and image-based product detection. It features:
- A Flask backend with modular endpoints and services
- A Next.js frontend for user interaction
- Jupyter notebooks for data science and model development
- Integration with Pinecone for vector search
- End-to-end workflow from data cleaning to model inference
- Modular pipeline system for easy model training and experimentation

---

## Table of Contents
- [Project Structure](#project-structure)
- [How Each Requirement is Implemented](#how-each-requirement-is-implemented)
- [Modular Pipeline System](#modular-pipeline-system)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [Notebook Execution Order](#notebook-execution-order)
- [API Endpoints & Frontend Pages](#api-endpoints--frontend-pages)
- [Usage Examples](#usage-examples)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Project Structure
```
ds_test/
  app.py                # Flask app entrypoint
  example_pipeline_usage.py  # Example script for modular pipeline
  src/
    endpoints/          # API blueprints (routes.py, image_detection.py)
    error_handlers.py   # Error handling utilities
    initialization.py   # Service/model/ocr/thread pool initialization
    models/             # Model definitions (cnn_model.py)
    pipeline/           # Data pipeline modules (data_ingestion.py, etc.)
    services/           # Business logic (recommendation_service.py)
    utils/              # Utilities (data_cleaning.py, logging_utils.py, etc.)
  test/                 # Unit/integration tests
  notebooks/            # Jupyter notebooks for data prep, OCR, training, etc.
  frontend/app/         # Next.js frontend
  requirements.txt      # Python dependencies
  ...
```

---

## How Each Requirement is Implemented
| Requirement                | Where Implemented                                      | How to Run/Test                                 | What to Expect                       |
|----------------------------|--------------------------------------------------------|-------------------------------------------------|--------------------------------------|
| Data Cleaning              | `src/utils/data_cleaning.py`, `notebooks/1_data_prep.ipynb` | Run notebook or import function                 | Cleaned CSV, logs                    |
| Vector DB                  | `src/utils/vector_db_utils.py`, `src/initialization.py`     | Start backend, check logs                       | Pinecone index created               |
| Product Recommendation     | `src/endpoints/routes.py`                                   | POST `/api/v1/recommendations`                  | JSON with products                   |
| OCR                        | `src/utils/handwriting_ocr.py`, `src/initialization.py`    | POST `/api/v1/ocr-query`                        | JSON with OCR result                 |
| Web Scraping               | `src/scripts/web_scraping_fix.py`, `notebooks/1_scrape_images.ipynb` | Run script or notebook              | Downloaded images                    |
| CNN Model                  | `src/models/cnn_model.py`, `notebooks/2_pipeline_integration_test.ipynb` | Run notebook or backend             | Model checkpoints, predictions       |
| Image Detection            | `src/endpoints/image_detection.py`                            | POST `/api/v1/product-detections`               | JSON with class, products            |
| Frontend                   | `frontend/app/`                                              | `npm run dev` in `frontend/app/`                | Web UI for queries                   |
| Tests                      | `test/`                                                     | `pytest` in project root                      | Test results                         |

---

## Modular Pipeline System

The project now includes a modular pipeline system that makes it easy to train and experiment with different model configurations. This system consists of two main classes:

### PipelineConfig
A configuration class that encapsulates all training parameters:

```python
from src.pipeline import PipelineConfig

# Basic configuration
config = PipelineConfig(
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    min_samples_per_class=10,
    num_workers=4,
    test_size=0.2
)

# Custom configuration for small datasets
small_config = PipelineConfig(
    batch_size=8,
    num_epochs=5,
    learning_rate=0.01,
    min_samples_per_class=2,
    num_workers=0,  # Avoid multiprocessing issues
    test_size=0.3
)
```

### Pipeline
The main pipeline class that handles the entire training workflow:

```python
from src.pipeline import Pipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(batch_size=16, num_epochs=5)
pipeline = Pipeline(config, project_root="./")

# Load and prepare data
data = pipeline.load_data()
train_loader, val_loader = pipeline.prepare_data_loaders()

# Create and train model
model = pipeline.create_model()
results = pipeline.train()
```

### Example Usage
See `example_pipeline_usage.py` for complete examples demonstrating:
- Different configuration scenarios
- Error handling
- Logging integration
- Training workflow

### Running the Example
```bash
python example_pipeline_usage.py
```

### Testing the Pipeline
```bash
pytest test/test_pipeline.py -v
```

The modular pipeline system provides:
- **Flexibility**: Easy to experiment with different configurations
- **Reproducibility**: Consistent training workflows
- **Logging**: Comprehensive logging for debugging and monitoring
- **Testing**: Full test coverage for all pipeline components
- **Error Handling**: Robust error handling and validation

---

## Setup Instructions
### Prerequisites
- Python 3.11 or 3.13 (project tested on both; recommend <=3.13 for best compatibility)
- Node.js & npm
- Jupyter for notebooks

### 1. Clone the Repository
```bash
git clone <repo-url>
cd ds_test
```

### 2. Backend Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend/app
npm install
```

---

## Environment Variables
Create a `.env` file in the project root with the following (example values):
```
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
```
**Required variables:**
- `PINECONE_API_KEY`: For vector DB integration
- `OPENAI_API_KEY`: For language model features (if used)


Create a `.env.local` file in `frontend/app` with the following:
```
NEXT_PUBLIC_BACKEND_URL=your_backend_url
```
**Required variables:**
- `NEXT_PUBLIC_BACKEND_URL` = http://127.0.0.1:5000

---

## Running the Project
### Start Backend (Flask)
```bash
python app.py
```
- The backend will be available at `http://localhost:5000/`

### Start Frontend (Next.js)
```bash
cd frontend/app
npm run dev
```
- The frontend will be available at `http://localhost:3000/`

### Run Tests
```bash
pytest
```

---

## Notebook Execution Order
The notebooks should be executed in the following order for proper data flow:

1. **`1_scrape_images.ipynb`** - Web scraping for product images
   - Downloads images from Amazon based on product descriptions
   - Creates the image dataset for training

2. **`2_pipeline_integration_test.ipynb`** - Pipeline integration and model training
   - Tests the complete data pipeline
   - Trains the CNN model
   - Validates model performance

3. **`3_model_performance_metrics.ipynb`** - Model evaluation and metrics
   - Evaluates trained model performance
   - Generates accuracy, F1 score, confusion matrix
   - Creates performance visualizations

4. **`4_ocr_utility.ipynb`** - OCR functionality testing
   - Tests handwriting OCR capabilities
   - Validates OCR accuracy on test images

5. **`5_similarity_analysis.ipynb`** - Similarity analysis and recommendations
   - Analyzes product similarities
   - Tests recommendation algorithms
   - Validates vector search functionality

**Note**: Each notebook builds upon the previous ones, so execute them in order for best results.

---

## API Endpoints & Frontend Pages
### API Endpoints
- `POST /api/v1/recommendations` — Product recommendation from text query
- `POST /api/v1/ocr-query` — OCR-based query (image upload)
- `POST /api/v1/product-detections` — Image-based product detection
- `GET /api/v1/health` — Health check

### Frontend Pages
- **Text Query Interface:** Submit text queries, view recommendations
- **Image Query Interface:** Upload handwritten queries, view results
- **Product Image Upload:** Upload product images, view detected class and recommendations

---

## Usage Examples
### Example: Product Recommendation (Text Query)
```bash
curl -X POST http://localhost:5000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"query": "laptop", "top_k": 3}'
```

### Example: OCR Query (Image Upload)
```bash
curl -X POST http://localhost:5000/api/v1/ocr-query \
  -F "file=@path_to_image.jpg"
```

### Example: Product Detection (Image Upload)
```bash
curl -X POST http://localhost:5000/api/v1/product-detections \
  -F "image=@path_to_product_image.jpg"
```

---

## Troubleshooting & FAQ
- **ModuleNotFoundError:** Ensure you are in the correct directory and your virtual environment is activated.
- **.env not loaded:** Double-check your `.env` file location and variable names.
- **Frontend/Backend not connecting:** Make sure both are running and CORS is enabled in Flask.
- **Pinecone/OpenAI errors:** Check your API keys and network connection.
- **Tests failing:** Run `pytest -v` for more detailed output.
- **Image loading errors:** Ensure the static/images directories exist and contain the scraped images.
- **Notebook path errors:** Make sure to run notebooks from the project root directory.

---

## Data & Model Files

- **Label Mapping Files:**
  - `src/data/dataset/stock_code_mapping.json`: Maps model output indices to stock codes (used for inference and result mapping).
  - `src/data/dataset/stock_code_to_product.json`: Maps stock codes to human-readable product names (used for display and recommendations).
  - These files ensure that model predictions are correctly mapped to product identities and descriptions at inference time.


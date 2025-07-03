# Product Recommendation, OCR, and Image-Based Detection System

## Overview
This project is a modular system for product recommendation, OCR-based query processing, and image-based product detection. It features:
- A Flask backend with modular endpoints and services
- A Next.js frontend for user interaction
- Jupyter notebooks for data science and model development
- Integration with Pinecone for vector search
- End-to-end workflow from data cleaning to model inference

---

## Table of Contents
- [Project Structure](#project-structure)
- [How Each Requirement is Implemented](#how-each-requirement-is-implemented)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Running the Project](#running-the-project)
- [API Endpoints & Frontend Pages](#api-endpoints--frontend-pages)
- [Usage Examples](#usage-examples)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Project Structure
```
ds_task_1ab/
  app.py                # Flask app entrypoint
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
| Web Scraping               | `src/scripts/web_scraping_fix.py`, `notebooks/2_ocr_web_scraping.ipynb` | Run script or notebook              | Downloaded images                    |
| CNN Model                  | `src/models/cnn_model.py`, `notebooks/3_cnn_model_training.ipynb` | Run notebook or backend             | Model checkpoints, predictions       |
| Image Detection            | `src/endpoints/image_detection.py`                            | POST `/api/v1/product-detections`               | JSON with class, products            |
| Frontend                   | `frontend/app/`                                              | `npm run dev` in `frontend/app/`                | Web UI for queries                   |
| Tests                      | `test/`                                                     | `pytest` in `ds_task_1ab/`                      | Test results                         |

---

## Setup Instructions
### Prerequisites
- Python <=3.11
- Node.js & npm
- Jupyter for notebooks

### 1. Clone the Repository
```bash
git clone <repo-url>
cd ds_task_1ab
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
Create a `.env` file in `ds_task_1ab/` with the following (example values):
```
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
```
**Required variables:**
- `PINECONE_API_KEY`: For vector DB integration
- `OPENAI_API_KEY`: For language model features (if used)

---

## Running the Project
### Start Backend (Flask)
```bash
cd ds_task_1ab
python app.py
```
- The backend will be available at `http://localhost:5000/`

### Start Frontend (Next.js)
```bash
cd frontend/app
npm run dev
```
- The frontend will be available at `http://localhost:3000/`

### Run Notebooks
```bash
cd ds_task_1ab/notebooks
jupyter notebook
```
- Open the desired notebook and run cells as needed.

### Run Tests
```bash
cd ds_task_1ab
pytest
```

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
  -F "file=@path_to_product_image.jpg"
```

---

## Troubleshooting & FAQ
- **ModuleNotFoundError:** Ensure you are in the correct directory and your virtual environment is activated.
- **.env not loaded:** Double-check your `.env` file location and variable names.
- **Frontend/Backend not connecting:** Make sure both are running and CORS is enabled in Flask.
- **Pinecone/OpenAI errors:** Check your API keys and network connection.
- **Tests failing:** Run `pytest -v` for more detailed output.


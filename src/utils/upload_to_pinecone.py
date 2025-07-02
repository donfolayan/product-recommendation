from pinecone import Pinecone, ServerlessSpec  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from dotenv import load_dotenv
import os
import time
import pandas as pd  # type: ignore
from pathlib import Path
from typing import Any

# Set up paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROGRESS_DIR = DATA_DIR / 'progress'
FAILED_RECORDS_DIR = DATA_DIR / 'failed_records'
PROGRESS_FILE = PROGRESS_DIR / 'upload_progress.csv'

# Create directories
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
FAILED_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
index_name = "product-vectors"

# Check if index exists, if not create it
if not pc.has_index(index_name):
    print("Creating new Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Using a standard embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully")
else:
    print("Index already exists")

# Get the index object
index = pc.Index(index_name)

# Initialize the sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully")

# Load or create progress tracking
if PROGRESS_FILE.exists():
    uploaded_ids = pd.read_csv(PROGRESS_FILE)['StockCode'].tolist()  # type: ignore
    print(f"Found {len(uploaded_ids)} previously uploaded records")
else:
    uploaded_ids = []
    print("Starting fresh upload")

# Process data in chunks
df = pd.read_csv(DATA_DIR / "dataset" / "cleaned_ecommerce_data.csv")
CHUNK_SIZE = 100
total_records = len(df)
print(f"Total records to process: {total_records}")

# Process data in chunks
for start_idx in range(0, total_records, CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, total_records)
    print(f"\nProcessing records {start_idx} to {end_idx}...")
    print(f"Progress: {end_idx/total_records*100:.2f}%")
    
    # Get chunk of data
    chunk_df = df.iloc[start_idx:end_idx]
    
    # Filter out already uploaded records
    chunk_df = chunk_df[~chunk_df['StockCode'].isin(uploaded_ids)]  # type: ignore
    
    if len(chunk_df) == 0:
        print("All records in this chunk already uploaded, skipping...")
        continue
    
    # Prepare data for this chunk
    vectors: list[dict[str, Any]] = []
    for _, row in chunk_df.iterrows():
        try:
            # Create a text representation of the product
            product_text = f"{row['Description']} {row['Country']}"  # type: ignore
            
            # Generate embedding for the product text
            embedding = model.encode(product_text)  # type: ignore
            
            # Create the vector record
            vector = {
                "id": str(row['StockCode']),  # type: ignore
                "values": embedding.tolist(),  # type: ignore
                "metadata": {
                    "InvoiceNo": str(row['InvoiceNo']),  # type: ignore
                    "CustomerID": str(row['CustomerID']),  # type: ignore
                    "Quantity": float(row['Quantity']),  # type: ignore
                    "UnitPrice": float(row['UnitPrice']),  # type: ignore
                    "Description": str(row['Description']),  # type: ignore
                    "Country": str(row['Country'])  # type: ignore
                }
            }
            vectors.append(vector)
        except Exception as e:
            print(f"Error processing record {row['StockCode']}: {str(e)}")
            continue
    
    # Upsert this chunk
    if vectors:  # Only try to upsert if we have valid vectors
        print("Uploading chunk to Pinecone...")
        try:
            # Upsert the vectors directly
            index.upsert(vectors=vectors)
            print(f"Successfully uploaded chunk {start_idx//CHUNK_SIZE + 1}")
            
            # Update progress tracking
            successful_ids = [v['id'] for v in vectors]
            uploaded_ids.extend(successful_ids)
            pd.DataFrame({'StockCode': uploaded_ids}).to_csv(PROGRESS_FILE, index=False)
            
        except Exception as e:
            print(f"Error uploading chunk: {str(e)}")
            # Save failed records for retry
            failed_records = pd.DataFrame([{
                'StockCode': v['id'],
                'InvoiceNo': v['metadata']['InvoiceNo'],
                'CustomerID': v['metadata']['CustomerID'],
                'Quantity': v['metadata']['Quantity'],
                'UnitPrice': v['metadata']['UnitPrice'],
                'Description': v['metadata']['Description'],
                'Country': v['metadata']['Country']
            } for v in vectors])
            failed_records.to_csv(FAILED_RECORDS_DIR / f'chunk_{start_idx}.csv', index=False)
            print(f"Saved failed records to {FAILED_RECORDS_DIR / f'chunk_{start_idx}.csv'}")
    
    # Add a small delay between chunks to prevent rate limiting
    time.sleep(1)

print("\nProcessing complete!")
print(f"Processed {total_records} records in chunks of {CHUNK_SIZE}")

# Check for any failed records
failed_files = list(FAILED_RECORDS_DIR.glob('*.csv'))
if failed_files:
    print(f"\nWarning: {len(failed_files)} chunks failed to upload. Check {FAILED_RECORDS_DIR} for details.")
else:
    print("\nAll chunks processed successfully!")
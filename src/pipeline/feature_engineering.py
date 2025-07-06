import pandas as pd
from typing import List, Any
from src.utils.vector_db_utils import VectorDBManager


def generate_embeddings(df: pd.DataFrame, vector_db_manager: VectorDBManager) -> List[Any]:
    """
    Generate feature embeddings for each row in the DataFrame using the provided VectorDBManager.
    Args:
        df (pd.DataFrame): Cleaned input DataFrame.
        vector_db_manager (VectorDBManager): Initialized vector DB manager for embedding.
    Returns:
        List[Any]: List of embedding vectors.
    """
    texts = [f"{row['Description']} {row['Country']}" for _, row in df.iterrows()]
    print(f"Generating embeddings for {len(texts)} rows (batched)...")
    embeddings = vector_db_manager.model.encode(
        texts,
        show_progress_bar=True
    )
    print("Embedding generation complete.")
    return embeddings

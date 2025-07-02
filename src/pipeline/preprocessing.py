import pandas as pd
from src.utils.data_cleaning import clean_dataset

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the input DataFrame using existing utilities.
    Args:
        df (pd.DataFrame): Raw input DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    return clean_dataset(df)


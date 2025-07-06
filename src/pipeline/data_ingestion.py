import os
import glob
import pandas as pd
from typing import Optional

def load_csv(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Args:
        filepath (str): Path to the CSV file.
        nrows (Optional[int]): Number of rows to read (for sampling/testing).
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath, nrows=nrows)

def build_image_label_df(images_root: str, stock_code_to_label: dict) -> pd.DataFrame:
    """
    Scan the images_root directory and build a DataFrame with image paths and class indices.
    Args:
        images_root (str): Path to the root directory containing StockCode-named folders of images.
        stock_code_to_label (dict): Mapping from StockCode (str) to class index (int).
    Returns:
        pd.DataFrame: DataFrame with columns 'image_path' and 'label'.
    """
    image_paths = []
    labels = []
    for stockcode_folder in os.listdir(images_root):
        folder_path = os.path.join(images_root, stockcode_folder)
        if os.path.isdir(folder_path) and stockcode_folder in stock_code_to_label:
            class_idx = stock_code_to_label[stockcode_folder]
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for img_file in glob.glob(os.path.join(folder_path, ext)):
                    image_paths.append(img_file)
                    labels.append(class_idx)
    return pd.DataFrame({'image_path': image_paths, 'label': labels})

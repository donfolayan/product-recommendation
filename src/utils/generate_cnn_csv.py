import os
import csv
import json
from pathlib import Path

def find_project_root(marker="requirements.txt"):
    p = Path.cwd()
    while not (p / marker).exists() and p != p.parent:
        p = p.parent
    if (p / marker).exists():
        return p
    raise FileNotFoundError(f"Could not find project root with marker '{marker}' from {Path.cwd()}")


def generate_final_cnn_training_data(project_root=None):
    """
    Generates the final_cnn_training_data.csv using only cleaned StockCodes and relative image paths.
    If project_root is not provided, it will be auto-detected using find_project_root().
    """
    if project_root is None:
        project_root = find_project_root()
    # Paths
    IMAGES_DIR = project_root / 'static' / 'images'
    LABEL_MAP_PATH = project_root / 'src' / 'data' / 'label_mapping.json'
    OUTPUT_CSV = project_root / 'src' / 'data' / 'final_cnn_training_data.csv'

    # Load label mapping
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    # Build a mapping from StockCode to integer string key
    stock_code_dirs = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
    stock_codes = sorted([d.name for d in stock_code_dirs])
    stockcode_to_index = {code: str(i) for i, code in enumerate(stock_codes)}

    # Save the stockcode_to_index mapping for later use
    STOCKCODE_TO_INDEX_PATH = project_root / 'models' / 'stockcode_to_index.json'
    STOCKCODE_TO_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STOCKCODE_TO_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(stockcode_to_index, f, indent=2)

    rows = []
    if not IMAGES_DIR.exists():
        print(f"Warning: Images directory does not exist: {IMAGES_DIR}")
        return
    for stock_code_dir in stock_code_dirs:
        stock_code = stock_code_dir.name
        index_key = stockcode_to_index.get(stock_code)
        label = index_key  # Use the integer index as the label
        if label is None:
            continue  # Skip if no index for this stock code
        for img_path in stock_code_dir.glob('*.jpg'):
            rel_path = os.path.relpath(img_path, project_root)
            rows.append({'image_path': rel_path, 'label': label})

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['image_path', 'label'])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    generate_final_cnn_training_data() 
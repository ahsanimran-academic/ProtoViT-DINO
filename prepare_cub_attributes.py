# prepare_cub_attributes.py (Final Corrected Version)
import os
import json
import pandas as pd
from tqdm import tqdm
import csv

def generate_attribute_json(cub_root_path: str, output_path: str):
    """
    Parses CUB attribute files and creates a JSON mapping image filenames
    to their 312-dimensional binary attribute vectors. This version uses a
    robust parser to handle potential file corruption.
    """
    print("Starting CUB attribute preparation (Robust Mode)...")

    # Define paths to necessary files
    images_txt_path = os.path.join(cub_root_path, 'images.txt')
    attributes_txt_path = os.path.join(cub_root_path, 'attributes', 'image_attribute_labels.txt')
    
    if not all(os.path.exists(p) for p in [images_txt_path, attributes_txt_path]):
        raise FileNotFoundError("CUB data files not found.")

    # 1. Read image IDs and filenames using Pandas (this file is usually clean)
    images_df = pd.read_csv(images_txt_path, sep=' ', names=['image_id', 'filepath'])
    # <<< FIX: Corrected typo from it_tuples to itertuples
    image_id_to_filename = {row.image_id: os.path.basename(row.filepath) for row in images_df.itertuples()}
    num_images = len(image_id_to_filename)
    print(f"Found {num_images} images.")

    # 2. Create the main data structure
    NUM_ATTRIBUTES = 312
    attributes_data = {
        filename: [0] * NUM_ATTRIBUTES for filename in image_id_to_filename.values()
    }

    # 3. Read the problematic attribute file line-by-line
    print(f"Robustly reading attributes from: {attributes_txt_path}")
    lines_processed = 0
    errors_skipped = 0
    with open(attributes_txt_path, 'r') as f:
        # Use a list comprehension with splitting for speed and simplicity
        lines = [line.strip().split() for line in f]

    for line in tqdm(lines, desc="Processing attributes"):
        lines_processed += 1
        try:
            # We only need the first 3 values: image_id, attr_id, is_present
            if len(line) < 3:
                errors_skipped += 1
                continue

            image_id = int(line[0])
            attr_id = int(line[1])
            is_present = int(line[2])

            if is_present == 1:
                filename = image_id_to_filename.get(image_id)
                if filename:
                    # attribute_id is 1-based, so subtract 1 for list index
                    attribute_index = attr_id - 1
                    if 0 <= attribute_index < NUM_ATTRIBUTES:
                        attributes_data[filename][attribute_index] = 1
        except (ValueError, IndexError):
            # This will catch lines that don't have enough columns or have non-integer values
            errors_skipped += 1
            continue
    
    print(f"Processed {lines_processed} lines. Skipped {errors_skipped} malformed lines.")

    # 4. Save to JSON
    print(f"Saving attribute data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(attributes_data, f)

    print("Attribute preparation complete!")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Ensure this path points to the root of your CUB_200_2011 dataset directory
    CUB_DATASET_ROOT = './datasets/CUB_200_2011/'
    
    # This is where the output file will be saved
    OUTPUT_JSON_PATH = './datasets/cub_attributes.json'
    
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    generate_attribute_json(CUB_DATASET_ROOT, OUTPUT_JSON_PATH)
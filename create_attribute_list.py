# create_attribute_list.py
import os

def generate_attribute_names_file(cub_root_path, output_path):
    """ Parses attributes.txt to create a clean list of attribute names. """
    attributes_file = os.path.join(cub_root_path, 'attributes.txt')
    if not os.path.exists(attributes_file):
        raise FileNotFoundError(f"Could not find attributes.txt in {cub_root_path}")

    with open(attributes_file, 'r') as f:
        lines = f.readlines()

    # The attribute name is the second part of the line, e.g., "1 has_bill_shape::curved"
    # We clean it up for better display.
    attribute_names = [line.strip().split(' ', 1)[1].replace('has_', '').replace('::', ': ') for line in lines]
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(attribute_names))
        
    print(f"Successfully created attribute names file at {output_path}")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Change this path to the root of your CUB_200_2011 dataset directory
    CUB_DATASET_ROOT = './datasets/'
    
    OUTPUT_NAMES_PATH = './datasets/cub_attribute_names.txt'
    
    generate_attribute_names_file(CUB_DATASET_ROOT, OUTPUT_NAMES_PATH)
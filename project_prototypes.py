# project_prototypes.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import os
import cv2
from PIL import Image

# --- We need to borrow from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/student_epoch_100.pth', # IMPORTANT: Set this to your trained model
    'cub_train_dir': './datasets/cub200_cropped/train_cropped', # IMPORTANT: Path to the training set to search for exemplars
    'output_dir': './prototype_projections/', # IMPORTANT: Where to save the prototype images
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 384, 4), # M, d, K
    'radius': 1,
    'sig_temp': 100.0,

    # --- PROJECTION PARAMETERS ---
    'batch_size': 128,
    'patch_size': 16, # Patch size of the ViT model
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 1. ## LOAD THE TRAINED STUDENT MODEL ##
    print("Loading pretrained student model...")
    vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
    model = ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape'], radius=config['radius'], sig_temp=config['sig_temp']).to(device)
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()
    print("Student model loaded.")

    # 2. ## PREPARE THE "PUSH" DATASET ##
    # Must use the training set. No data augmentation, shuffle=False.
    push_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    push_dataset = ImageFolder(root=config['cub_train_dir'], transform=push_transform)
    push_loader = DataLoader(push_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # Get a list of all image paths, crucial for saving results
    image_paths = [s[0] for s in push_dataset.samples]

    # 3. ## INITIALIZE PROTOTYPE BANK ##
    # We need to store the best activation score and location for each sub-prototype
    num_prototypes = config['prototype_shape'][0]
    num_sub_prototypes = config['prototype_shape'][2]
    
    # Initialize with -infinity because we want to find the maximum similarity
    best_scores = np.full((num_prototypes, num_sub_prototypes), -np.inf, dtype=np.float64)
    
    # This dictionary will store {'image_index': ..., 'patch_index': ...}
    best_locations = {}

    # 4. ## FIND THE BEST ACTIVATING PATCHES ##
    print("Finding best activating patches for each prototype...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(push_loader, desc="Pushing Prototypes")):
            images = images.to(device)
            
            # Use our new push_forward method
            _, values, indices = model.push_forward(images)
            
            # Iterate over each item in the batch
            for batch_idx in range(images.size(0)):
                # Iterate over each prototype
                for proto_j in range(num_prototypes):
                    # Iterate over each sub-prototype
                    for sub_k in range(num_sub_prototypes):
                        score = values[batch_idx, proto_j, sub_k].item()
                        
                        # If this is the best score we've seen for this sub-prototype
                        if score > best_scores[proto_j, sub_k]:
                            best_scores[proto_j, sub_k] = score
                            best_locations[(proto_j, sub_k)] = {
                                'image_index': i * config['batch_size'] + batch_idx,
                                'patch_index': indices[batch_idx, proto_j, sub_k].item()
                            }

    # 5. ## SAVE THE PROTOTYPE IMAGE EXEMPLARS ##
    print("Saving prototype image exemplars...")
    grid_size = config['img_size'] // config['patch_size'] # e.g., 224 // 16 = 14

    for (proto_j, sub_k), location in tqdm(best_locations.items(), desc="Saving Images"):
        # Open the original, un-normalized image
        original_img_path = image_paths[location['image_index']]
        img = Image.open(original_img_path).convert("RGB")
        img = img.resize((config['img_size'], config['img_size']))
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Calculate patch coordinates
        patch_idx = location['patch_index']
        patch_row = patch_idx // grid_size
        patch_col = patch_idx % grid_size
        
        x1 = patch_col * config['patch_size']
        y1 = patch_row * config['patch_size']
        x2 = x1 + config['patch_size']
        y2 = y1 + config['patch_size']
        
        # Crop the patch
        patch_img = img_bgr[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(config['output_dir'], f'prototype-j{proto_j}-k{sub_k}.png'), patch_img)
        
        # Save the original image with the patch highlighted
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(config['output_dir'], f'prototype-highlighted-j{proto_j}-k{sub_k}.png'), img_bgr)
        
    print(f"Prototype projections saved to: {config['output_dir']}")

if __name__ == '__main__':
    main()
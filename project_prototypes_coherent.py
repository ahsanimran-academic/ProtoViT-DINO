# project_prototypes_coherent.py
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
    'student_checkpoint_path': './saved_models/protovit_dino/best_model.pth', 
    'cub_train_dir': './datasets/cub200_cropped/train_cropped/', 
    'output_dir': './prototype_projections_coherent/', # Save to a new directory
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 768, 4), 
    
    # --- PROJECTION PARAMETERS ---
    'batch_size': 128,
    'patch_size': 16,
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 1. ## LOAD THE TRAINED STUDENT MODEL ##
    print("Loading pretrained student model...")
    vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
    model = ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape']).to(device)
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()
    print("Student model loaded.")

    # 2. ## PREPARE THE "PUSH" DATASET ##
    push_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    push_dataset = ImageFolder(root=config['cub_train_dir'], transform=push_transform)
    push_loader = DataLoader(push_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    image_paths = [s[0] for s in push_dataset.samples]

    # 3. ## FIND THE BEST IMAGE FOR EACH MAIN PROTOTYPE ##
    print("Finding the best image for each main prototype...")
    num_prototypes = config['prototype_shape'][0]
    best_scores = np.full(num_prototypes, -np.inf, dtype=np.float64)
    best_image_indices = np.zeros(num_prototypes, dtype=np.int64)

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(push_loader, desc="Finding Best Images")):
            images = images.to(device)
            # Get the aggregated scores for each main prototype
            proto_scores, _, _ = model.greedy_distance(images) # Shape: [batch_size, 256]
            
            # Find the max score for each prototype across the batch
            batch_max_scores, batch_argmax_indices = torch.max(proto_scores, dim=0)
            
            # Check if any scores in this batch are better than the global best
            for j in range(num_prototypes):
                if batch_max_scores[j] > best_scores[j]:
                    best_scores[j] = batch_max_scores[j].item()
                    best_image_indices[j] = i * config['batch_size'] + batch_argmax_indices[j].item()

    # 4. ## SAVE THE COHERENT PROTOTYPE VISUALIZATIONS ##
    print("Saving coherent prototype visualizations...")
    grid_size = config['img_size'] // config['patch_size']
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)] # Yellow, Red, Green, Blue

    with torch.no_grad():
        for j in tqdm(range(num_prototypes), desc="Saving Images"):
            # Get the single best image for this prototype
            best_img_idx = best_image_indices[j]
            img_tensor, _ = push_dataset[best_img_idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Re-run the push_forward pass on ONLY this image to get patch locations
            _, _, indices = model.push_forward(img_tensor)
            # indices shape: [1, num_prototypes, num_sub_prototypes]
            sub_prototype_patch_indices = indices[0, j, :].cpu().numpy()

            # Open the original image
            original_img_path = image_paths[best_img_idx]
            img = Image.open(original_img_path).convert("RGB").resize((config['img_size'], config['img_size']))
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Draw all 4 bounding boxes on this single image
            for k in range(config['prototype_shape'][2]):
                patch_idx = sub_prototype_patch_indices[k]
                patch_row = patch_idx // grid_size
                patch_col = patch_idx % grid_size
                x1, y1 = patch_col * config['patch_size'], patch_row * config['patch_size']
                x2, y2 = x1 + config['patch_size'], y1 + config['patch_size']
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), colors[k], 2)
            
            # Save the composite image
            cv2.imwrite(os.path.join(config['output_dir'], f'prototype-composition-j{j}.png'), img_bgr)

    print(f"Coherent prototype visualizations saved to: {config['output_dir']}")

if __name__ == '__main__':
    main()
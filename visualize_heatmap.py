# visualize_heatmap.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/student_epoch_10.pth',
    'cub_train_dir': './datasets/cub200_cropped/train_cropped/',
    'output_file': './class_prototype_heatmap.png',
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 384, 4),
    'num_classes': 200,
    'batch_size': 128,
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. ## LOAD MODEL AND DATA ##
    print("Loading model and data...")
    vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
    model = ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape']).to(device)
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(root=config['cub_train_dir'], transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # 2. ## COMPUTE AVERAGE ACTIVATIONS ##
    print("Computing average prototype activations per class...")
    num_prototypes = config['prototype_shape'][0]
    # Store activations for each class
    class_activations = [[] for _ in range(config['num_classes'])]

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Calculating Activations"):
            images = images.to(device)
            _, proto_scores = model(images)
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_activations[label].append(proto_scores[i].cpu().numpy())

    # Average the activations for each class
    mean_activations = np.zeros((config['num_classes'], num_prototypes))
    for i in range(config['num_classes']):
        if class_activations[i]:
            mean_activations[i, :] = np.mean(np.array(class_activations[i]), axis=0)

    # 3. ## PLOT THE CLUSTERMAP ##
    print("Generating clustermap...")
    # Normalize the data for better visualization (e.g., scale each prototype's activations across classes)
    df = pd.DataFrame(mean_activations)
    df_normalized = (df - df.mean()) / (df.std() + 1e-6) # Z-score normalization

    # Create the clustermap
    g = sns.clustermap(
        df_normalized,
        figsize=(20, 15),
        cmap='viridis', # A good colormap for this
        xticklabels=False, # Too many prototypes to label
        yticklabels=False  # Too many classes to label
    )
    
    g.fig.suptitle('Clustered Heatmap of Mean Prototype Activations per Class', fontsize=16)
    plt.savefig(config['output_file'], bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to {config['output_file']}")
    plt.close()

if __name__ == '__main__':
    main()
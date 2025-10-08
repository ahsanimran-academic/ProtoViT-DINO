# evaluate_mlp_probe.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/best_model.pth',
    'cub_train_dir': './datasets/cub200_cropped/train_cropped', # IMPORTANT: Path to labeled training set
    'cub_test_dir': './datasets/cub200_cropped/test_cropped',   # IMPORTANT: Path to labeled test set
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 768, 4), 
    
    # --- MLP PROBE TRAINING ---
    'num_classes': 200,
    'batch_size': 128,
    'epochs': 50, # MLP may converge faster
    'learning_rate': 1e-3, # Use a standard LR for Adam
    'weight_decay': 1e-4,
    
    'feature_source': 'proto_scores' 
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. ## LOAD AND FREEZE THE TRAINED STUDENT MODEL ##
    print("Loading and freezing student model...")
    vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
    model = ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape']).to(device)
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Student model loaded and frozen.")

    # 2. ## PREPARE THE LABELED DATA ##
    # <<< FIX: Replaced placeholder code with the full implementation >>>
    eval_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=config['cub_train_dir'], transform=eval_transform)
    test_dataset = ImageFolder(root=config['cub_test_dir'], transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    print(f"Data loaded: {len(train_dataset)} train images, {len(test_dataset)} test images.")
    
    # 3. ## DEFINE THE NON-LINEAR (MLP) PROBE ##
    if config['feature_source'] == 'proto_scores':
        feature_dim = config['prototype_shape'][0]
    else:
        feature_dim = config['prototype_shape'][1]
    
    mlp_probe = nn.Sequential(
        nn.Linear(feature_dim, 1024),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(1024, config['num_classes'])
    ).to(device)
    print(f"Created MLP probe with input dimension {feature_dim}.")

    # 4. ## TRAIN THE MLP PROBE ##
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(mlp_probe.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    for epoch in range(config['epochs']):
        mlp_probe.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training MLP Probe Epoch {epoch+1}/{config['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                cls_token, proto_scores = model(images)
                features = proto_scores if config['feature_source'] == 'proto_scores' else cls_token

            outputs = mlp_probe(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} MLP Probe Loss: {running_loss / len(train_loader):.4f}")

    # 5. ## EVALUATE THE TRAINED MLP PROBE ##
    print("Evaluating MLP probe on the test set...")
    mlp_probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            cls_token, proto_scores = model(images)
            features = proto_scores if config['feature_source'] == 'proto_scores' else cls_token
            
            outputs = mlp_probe(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print(f"🔬 MLP Probe Final Accuracy: {accuracy:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
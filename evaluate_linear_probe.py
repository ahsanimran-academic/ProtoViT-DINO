# evaluate_linear_probe.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- We need to borrow from your existing model files ---
# This assumes your original model definition files are in the same directory
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/best_model.pth', # IMPORTANT: Set this to your trained model
    'cub_train_dir': './datasets/cub200_cropped/train_cropped', # IMPORTANT: Path to labeled training set
    'cub_test_dir': './datasets/cub200_cropped/test_cropped',   # IMPORTANT: Path to labeled test set

    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 768, 4), # M, d, K
    'radius': 1,
    'sig_temp': 100.0,
    
    # --- LINEAR PROBE TRAINING ---
    'num_classes': 200, # CUB-200-2011 has 200 species
    'batch_size': 128,
    'epochs': 200,
    'learning_rate': 0.01,
    'weight_decay': 1e-6,
    
    # --- FEATURE SELECTION ---
    # Choose 'proto_scores' or 'cls_token'
    'feature_source': 'proto_scores' 
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. ## LOAD THE TRAINED STUDENT MODEL ##
    print("Loading pretrained student model...")
    # Reconstruct the model architecture exactly as during training
    vit_features = construct_PPNet(
        base_architecture=config['base_architecture'],
        pretrained=True,
        img_size=config['img_size'],
        prototype_shape=config['prototype_shape'],
        num_classes=1
    ).features

    # Instantiate the base model (not the DINO wrapper)
    model = ProtoViTBase(
        features=vit_features,
        img_size=config['img_size'],
        prototype_shape=config['prototype_shape'],
        radius=config['radius'],
        sig_temp=config['sig_temp']
    ).to(device)

    # Load the saved weights from your SSL training
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    
    # --- CRITICAL: Freeze the backbone ---
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Student model loaded and frozen.")

    # 2. ## PREPARE THE LABELED DATA ##
    # Use standard evaluation transforms
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

    # 3. ## DEFINE THE LINEAR CLASSIFIER ##
    if config['feature_source'] == 'proto_scores':
        # Feature dimension is the number of prototypes (M)
        feature_dim = config['prototype_shape'][0]
    elif config['feature_source'] == 'cls_token':
        # Feature dimension is the ViT embedding dimension (d)
        feature_dim = config['prototype_shape'][1]
    else:
        raise ValueError("feature_source must be 'proto_scores' or 'cls_token'")

    linear_classifier = nn.Linear(feature_dim, config['num_classes']).to(device)
    print(f"Created linear classifier with input dimension {feature_dim}.")

    # 4. ## TRAIN THE LINEAR CLASSIFIER ##
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_classifier.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.9)
    
    for epoch in range(config['epochs']):
        linear_classifier.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                # Extract features from the frozen backbone
                cls_token, proto_scores = model(images)
                
                if config['feature_source'] == 'proto_scores':
                    features = proto_scores
                else: # cls_token
                    features = cls_token

            # Forward pass through the linear layer
            outputs = linear_classifier(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

    # 5. ## EVALUATE THE TRAINED CLASSIFIER ##
    print("Evaluating on the test set...")
    linear_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            # Extract features
            cls_token, proto_scores = model(images)
            if config['feature_source'] == 'proto_scores':
                features = proto_scores
            else: # cls_token
                features = cls_token

            # Get predictions
            outputs = linear_classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print(f"Linear Probe Final Accuracy: {accuracy:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
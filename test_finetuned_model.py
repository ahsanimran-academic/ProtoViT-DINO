# test_finetuned_model.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    # IMPORTANT: Point this to the fine-tuned model you saved
    'finetuned_model_path': './saved_models/protovit_dino/best_finetuned_model.pth', 
    'cub_test_dir': './datasets/cub200_cropped/test_cropped/',
    
    # --- MODEL ARCHITECTURE (CRITICAL: Must match the saved model) ---
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 768, 4),
    
    # --- EVALUATION PARAMETERS ---
    'num_classes': 200,
    'batch_size': 128,
    
    # --- FEATURE SELECTION (Must match how the model was fine-tuned) ---
    'feature_source': 'proto_scores' 
}

# --- A wrapper to combine the backbone and a new classifier head ---
# This must be defined to reconstruct the model structure before loading weights
class FullClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes, feature_source):
        super().__init__()
        self.backbone = backbone
        self.feature_source = feature_source
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        cls_token, proto_scores = self.backbone(x)
        features = proto_scores if self.feature_source == 'proto_scores' else cls_token
        return self.classifier(features)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. ## RECONSTRUCT THE MODEL ARCHITECTURE ##
    print("Reconstructing model architecture...")
    # First, build the backbone structure
    vit_features = construct_PPNet(
        base_architecture=config['base_architecture'], pretrained=False, # No need to download pretrained weights
        img_size=config['img_size'],
        prototype_shape=config['prototype_shape'], num_classes=1
    ).features
    backbone = ProtoViTBase(
        features=vit_features, img_size=config['img_size'], 
        prototype_shape=config['prototype_shape']
    )
    
    # Then, build the full classifier structure
    if config['feature_source'] == 'proto_scores':
        feature_dim = config['prototype_shape'][0]
    else: # 'cls_token'
        feature_dim = config['prototype_shape'][1]
    
    model = FullClassifier(backbone, feature_dim, config['num_classes'], config['feature_source']).to(device)

    # 2. ## LOAD THE SAVED FINE-TUNED WEIGHTS ##
    print(f"Loading fine-tuned weights from: {config['finetuned_model_path']}")
    model.load_state_dict(torch.load(config['finetuned_model_path'], map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 3. ## PREPARE THE TEST DATA ##
    test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolder(root=config['cub_test_dir'], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    print(f"Test data loaded: {len(test_dataset)} images.")

    # 4. ## EVALUATE ON THE TEST SET ##
    print("Evaluating on the test set...")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print(f"✅ Final Test Accuracy: {accuracy:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
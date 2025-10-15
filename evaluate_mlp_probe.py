# evaluate_mlp_probe.py (Upgraded with Validation and Early Stopping)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    'student_checkpoint_path': './saved_models/protovit_dino/best_model.pth',
    'cub_train_dir': './datasets/cub200_cropped/train_cropped/',
    'cub_test_dir': './datasets/cub200_cropped/test_cropped/',
    'save_dir': './saved_models/protovit_dino/',
    
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (512, 768, 4), 
    
    'num_classes': 200,
    'batch_size': 128,
    'epochs': 200, # Set a high number, the script will effectively "early stop" by using the best model
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'validation_split': 0.1, # Use 10% of training data for validation
    
    'feature_source': 'proto_scores' 
}

# --- A dedicated evaluation function ---
def evaluate(backbone, probe, dataloader, device, criterion, feature_source_cfg):
    backbone.eval()
    probe.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            cls_token, proto_scores = backbone(images)
            features = proto_scores if feature_source_cfg == 'proto_scores' else cls_token
            
            outputs = probe(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(config['save_dir'], exist_ok=True)

    # 1. ## LOAD AND FREEZE THE TRAINED STUDENT MODEL ##
    print("Loading and freezing student model...")
    vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
    model = ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape']).to(device)
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Student model loaded and frozen.")

    # 2. ## PREPARE DATA WITH VALIDATION SPLIT ##
    print("Preparing data with a validation split...")
    # Use the same simple transform for both train and val in a probe setting
    eval_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_train_dataset = ImageFolder(root=config['cub_train_dir'], transform=eval_transform)
    
    val_size = int(len(full_train_dataset) * config['validation_split'])
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    
    test_dataset = ImageFolder(root=config['cub_test_dir'], transform=eval_transform)
    
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    print(f"Data loaded: {len(train_subset)} train, {len(val_subset)} validation, {len(test_dataset)} test images.")

    # 3. ## DEFINE THE MLP PROBE ##
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

    # 4. ## TRAIN THE MLP PROBE WITH VALIDATION ##
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(mlp_probe.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    best_val_accuracy = 0.0
    best_probe_path = os.path.join(config['save_dir'], 'best_mlp_probe.pth')

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
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation step
        val_loss, val_accuracy = evaluate(model, mlp_probe, val_loader, device, criterion, config['feature_source'])
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # Save the best probe based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(mlp_probe.state_dict(), best_probe_path)
            print(f"*** New best MLP probe saved with Val Acc: {best_val_accuracy:.2f}% ***")

    # 5. ## FINAL EVALUATION USING THE BEST PROBE ##
    print("\n--- Final Evaluation on Test Set ---")
    print(f"Loading best MLP probe from {best_probe_path} (Val Acc: {best_val_accuracy:.2f}%)")
    mlp_probe.load_state_dict(torch.load(best_probe_path))
    
    test_loss, test_accuracy = evaluate(model, mlp_probe, test_loader, device, criterion, config['feature_source'])

    print("\n" + "="*50)
    print(f"🔬 MLP Probe Final Test Accuracy: {test_accuracy:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
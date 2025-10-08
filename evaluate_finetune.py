# evaluate_finetune.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os


# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/best_model.pth', # Path to your SSL-trained model
    'cub_train_dir': './datasets/cub200_cropped/train_cropped', 
    'cub_test_dir': './datasets/cub200_cropped/test_cropped',
    'save_dir': './saved_models/protovit_dino/', # Directory to save the final fine-tuned model
    'log_filename': 'finetune_log.txt', # <<< ADDED: Log file name >>>
    
    # --- MODEL ARCHITECTURE (CRITICAL: Must match the checkpoint from train.py) ---
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 768, 4), # M, D, K
    
    # --- FINE-TUNING PARAMETERS ---
    'num_classes': 200,
    'batch_size': 128, # Adjust based on your GPU memory
    'epochs': 300,
    'learning_rate': 1e-5, # Use a very small learning rate for fine-tuning
    'weight_decay': 0.01,
    'validation_split': 0.1, # Use 10% of training data for validation
    
    # --- FEATURE SELECTION ---
    'feature_source': 'proto_scores' 
}

# <<< ADDED: A simple logger to write to a file and console >>>
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write('--- Fine-Tuning Log ---\n')
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

# --- A wrapper to combine the backbone and a new classifier head ---
class FullClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes, feature_source):
        super().__init__()
        self.backbone = backbone
        self.feature_source = feature_source
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Get features from the backbone
        cls_token, proto_scores = self.backbone(x)
        # Select the feature source
        features = proto_scores if self.feature_source == 'proto_scores' else cls_token
        # Pass features to the new classifier
        return self.classifier(features)

# --- A dedicated evaluation function ---
def evaluate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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

    # <<< ADDED: Initialize the logger >>>
    logger = Logger(os.path.join(config['save_dir'], config['log_filename']))
    logger.log(f"Using device: {device}")

    # 1. ## LOAD THE PRE-TRAINED STUDENT BACKBONE ##
    print(f"Loading student backbone from: {config['student_checkpoint_path']}")
    vit_features = construct_PPNet(
        base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'],
        prototype_shape=config['prototype_shape'], num_classes=1
    ).features
    backbone = ProtoViTBase(
        features=vit_features, img_size=config['img_size'], 
        prototype_shape=config['prototype_shape']
    )
    backbone.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    print("Student backbone loaded.")

    # 2. ## BUILD THE FULL CLASSIFIER MODEL ##
    if config['feature_source'] == 'proto_scores':
        feature_dim = config['prototype_shape'][0]
    else: # 'cls_token'
        feature_dim = config['prototype_shape'][1]
    
    model = FullClassifier(backbone, feature_dim, config['num_classes'], config['feature_source']).to(device)

    # 3. ## PREPARE DATA WITH VALIDATION SPLIT ##
    print("Preparing data with a validation split...")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['img_size'], scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_train_dataset = ImageFolder(root=config['cub_train_dir'], transform=train_transform)
    val_size = int(len(full_train_dataset) * config['validation_split'])
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    val_subset.dataset = ImageFolder(root=config['cub_train_dir'], transform=val_test_transform)
    test_dataset = ImageFolder(root=config['cub_test_dir'], transform=val_test_transform)
    
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    print(f"Data loaded: {len(train_subset)} train, {len(val_subset)} validation, {len(test_dataset)} test images.")

    # 4. ## SET UP OPTIMIZER AND LOSS FOR FINE-TUNING ##
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_accuracy = 0.0
    best_model_path = os.path.join(config['save_dir'], 'best_finetuned_model.pth')

    # 5. ## FINE-TUNING LOOP with Validation ##
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Fine-Tuning Epoch {epoch+1}/{config['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # <<< CHANGED: Use logger.log to save epoch results >>>
        log_message = f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        logger.log(log_message)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved with Val Acc: {best_val_accuracy:.2f}% ***")
            # <<< CHANGED: Use logger.log to save best model info >>>
            logger.log(f"*** New best model saved with Val Acc: {best_val_accuracy:.2f}% ***")


    # 6. ## FINAL EVALUATION on Test Set ##
    print("\n--- Final Evaluation on Test Set ---")
    print(f"Loading best model from {best_model_path} (Val Acc: {best_val_accuracy:.2f}%)")
    # --- FINAL EVALUATION on Test Set ---
    logger.log("\n--- Final Evaluation on Test Set ---")
    logger.log(f"Loading best model from {best_model_path} (Val Acc: {best_val_accuracy:.2f}%)")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    test_loss, test_accuracy = evaluate(model, test_loader, device, criterion)
    
    print("\n" + "="*50)
    print(f"🔥 Fine-Tuning Final Test Accuracy: {test_accuracy:.2f}%")
    print("="*50)
    # <<< CHANGED: Use logger.log for final results >>>
    logger.log("\n" + "="*50)
    logger.log(f"🔥 Fine-Tuning Final Test Accuracy: {test_accuracy:.2f}%")
    logger.log("="*50)

if __name__ == '__main__':
    main()
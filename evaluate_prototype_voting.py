# evaluate_prototype_voting.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from collections import Counter

# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/best_model.pth',
    'cub_train_dir': './datasets/cub200_cropped/train_cropped/',
    'cub_test_dir': './datasets/cub200_cropped/test_cropped/',
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (512, 768, 4), # M, d, K
    
    # --- EVALUATION PARAMETERS ---
    'num_classes': 200,
    'batch_size': 128,
    'k_votes': 5 # The "Top-k" prototypes to use for voting
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    
    # We need the training set to build the prototype-to-class map
    train_dataset = ImageFolder(root=config['cub_train_dir'], transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    test_dataset = ImageFolder(root=config['cub_test_dir'], transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    print("Model and data loaded.")

    # 2. ## PHASE A: DETERMINE PROTOTYPE CLASS IDENTITY ##
    print("\n--- Phase A: Determining Prototype Class Identity from Training Set ---")
    num_prototypes = config['prototype_shape'][0]
    # For each prototype, we'll store a list of the classes of images that activated it most
    prototype_class_associations = [[] for _ in range(num_prototypes)]

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Analyzing Training Set"):
            images = images.to(device)
            _, proto_scores = model(images) # Get prototype activations
            
            # Find the top activating prototype for each image in the batch
            top_proto_indices = torch.argmax(proto_scores, dim=1)
            
            for i in range(len(images)):
                proto_idx = top_proto_indices[i].item()
                class_label = labels[i].item()
                prototype_class_associations[proto_idx].append(class_label)
    
    # Now, for each prototype, find the most common class associated with it
    prototype_to_class_map = {}
    for i in range(num_prototypes):
        if prototype_class_associations[i]:
            most_common_class = Counter(prototype_class_associations[i]).most_common(1)[0][0]
            prototype_to_class_map[i] = most_common_class
    
    print(f"Mapped {len(prototype_to_class_map)} out of {num_prototypes} prototypes to a majority class.")

    # 3. ## PHASE B: EVALUATE ON TEST SET WITH VOTING ##
    print(f"\n--- Phase B: Evaluating with Top-{config['k_votes']} Prototype Voting ---")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device)
            _, proto_scores = model(images)
            
            # Get top-k activating prototypes for each image
            top_k_scores, top_k_indices = torch.topk(proto_scores, k=config['k_votes'], dim=1)
            
            for i in range(len(images)):
                # Get the class "votes" from the top-k prototypes
                votes = [prototype_to_class_map.get(idx.item()) for idx in top_k_indices[i]]
                # Filter out any prototypes that didn't have a majority class
                valid_votes = [v for v in votes if v is not None]
                
                if not valid_votes:
                    # If no top prototypes had a clear class, we can't make a prediction.
                    # This is rare but possible. We'll count it as incorrect.
                    prediction = -1 
                else:
                    # The prediction is the most common vote
                    prediction = Counter(valid_votes).most_common(1)[0][0]

                if prediction == labels[i].item():
                    correct += 1
                total += 1

    # 4. ## REPORT RESULTS ##
    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print(f"Top-{config['k_votes']} Prototype Voting Accuracy: {accuracy:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()
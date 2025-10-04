# evaluate_attributes.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import json
from PIL import Image

# --- We need torchmetrics for mAP calculation ---
from torchmetrics.classification import MultilabelAveragePrecision

# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase, ProtoViT_DINO

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/student_epoch_100.pth',
    'cub_test_dir': './datasets/cub200_cropped/test_cropped',
    'attributes_json': './datasets/cub_attributes.json',
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 384, 4),
    'dino_out_dim': 4096,
    
    # --- EVALUATION PARAMETERS ---
    'num_attributes': 312,
    'batch_size': 128,
}

# --- A specific Dataset class for this evaluation ---
class AttributeTestDataset(ImageFolder):
    """
    Dataset for the test set that returns an image and its
    ground-truth 312-dimensional attribute vector.
    """
    def __init__(self, root, attributes_json_path, transform=None):
        super().__init__(root, transform=transform)
        
        print(f"Loading attributes from {attributes_json_path}")
        with open(attributes_json_path, 'r') as f:
            self.attributes_data = json.load(f)
            
    def __getitem__(self, index):
        path, _ = self.samples[index]
        filename = os.path.basename(path)
        sample = self.loader(path)
        
        # Standard transform
        if self.transform is not None:
            sample = self.transform(sample)
        
        # Get ground-truth attribute vector
        attribute_vector = self.attributes_data.get(filename, [0] * config['num_attributes'])
        attributes = torch.tensor(attribute_vector, dtype=torch.float32)
        
        return sample, attributes

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. ## LOAD THE FULL DINO MODEL AND STUDENT WEIGHTS ##
    print("Loading model...")
    def create_base_model():
        vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
        return ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape'])

    # Instantiate the full DINO model to get access to the attribute_head
    model = ProtoViT_DINO(
        base_model_fn=create_base_model,
        out_dim=config['dino_out_dim']
    ).to(device)

    # Load your trained student weights into the student part of the model
    model.student.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()
    print("Model loaded.")

    # 2. ## PREPARE THE TEST DATA ##
    eval_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = AttributeTestDataset(
        root=config['cub_test_dir'],
        attributes_json_path=config['attributes_json'],
        transform=eval_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    print(f"Test data loaded: {len(test_dataset)} images.")

    # 3. ## INITIALIZE METRIC ##
    # We use 'macro' average to treat each attribute equally
    metric = MultilabelAveragePrecision(num_labels=config['num_attributes'], average='macro').to(device)

    # 4. ## EVALUATION LOOP ##
    print("Evaluating attribute prediction performance...")
    with torch.no_grad():
        for images, true_attributes in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            true_attributes = true_attributes.to(device)
            
            # Get prototype scores from the student model
            _, proto_scores = model.student(images)
            
            # Get attribute predictions from the attribute_head
            attr_logits = model.attribute_head(proto_scores)
            
            # Update the metric with predictions and ground truth
            # torchmetrics handles the conversion from logits to probabilities
            metric.update(attr_logits, true_attributes.int())

    # 5. ## COMPUTE AND DISPLAY FINAL SCORE ##
    mAP_score = metric.compute()
    
    print("\n" + "="*50)
    print(f"Attribute Prediction mAP: {mAP_score.item():.4f}")
    print("="*50)
    print("(mAP is the mean Average Precision across all 312 attributes. Higher is better.)")

if __name__ == '__main__':
    main()
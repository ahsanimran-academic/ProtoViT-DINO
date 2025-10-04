# explain_image.py
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase, ProtoViT_DINO

# --- Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/student_epoch_100.pth', # IMPORTANT: Path to your trained model
    'attribute_names_path': './datasets/cub_attribute_names.txt', # Path to the file from Step 1
    'test_image_path': './datasets/cub200_cropped/test_cropped/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg', # An example test image
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 384, 4), # M, d, K
    'dino_out_dim': 4096,
    
    # --- EXPLANATION PARAMETERS ---
    'top_k_prototypes': 5, # How many top prototypes to show
    'attribute_threshold': 0.5, # Probability threshold to show a predicted attribute
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. ## LOAD ATTRIBUTE NAMES ##
    with open(config['attribute_names_path'], 'r') as f:
        attribute_names = [line.strip() for line in f.readlines()]

    # 2. ## LOAD THE FULL DINO MODEL AND STUDENT WEIGHTS ##
    print("Loading model...")
    # Define a function to create the base model, needed by the DINO wrapper
    def create_base_model():
        vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
        return ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape'])

    # Instantiate the full DINO model wrapper, which contains the attribute_head
    full_model = ProtoViT_DINO(
        base_model_fn=create_base_model,
        out_dim=config['dino_out_dim']
    ).to(device)

    # Load your trained student weights into the student part of the model
    full_model.student.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    full_model.eval()
    print("Model loaded.")

    # 3. ## PREPARE THE TEST IMAGE ##
    eval_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(config['test_image_path']).convert('RGB')
    img_tensor = eval_transform(img).unsqueeze(0).to(device) # Add batch dimension

    # 4. ## GENERATE THE EXPLANATION ##
    print(f"\n--- Generating Explanation for: {config['test_image_path']} ---")
    with torch.no_grad():
        # Get prototype scores from the student model
        _, proto_scores = full_model.student(img_tensor)
        
        # **NEW**: Pass scores through the attribute head to get predictions
        attr_logits = full_model.attribute_head(proto_scores)
        attr_probs = torch.sigmoid(attr_logits).squeeze(0) # Remove batch dim

    # --- Process and Display Results ---
    
    # A. Top-k Prototypes
    top_scores, top_indices = torch.topk(proto_scores.squeeze(0), config['top_k_prototypes'])
    print(f"\n## Top {config['top_k_prototypes']} Activating Prototypes ##")
    for i in range(config['top_k_prototypes']):
        print(f"  - Prototype #{top_indices[i].item()} (Activation Score: {top_scores[i]:.4f})")
    print("(To get visual exemplars and names, run project_prototypes.py and name_prototypes.py)")
    
    # B. Predicted Attributes
    predicted_attr_indices = (attr_probs > config['attribute_threshold']).nonzero(as_tuple=True)[0]
    print(f"\n## Predicted Attributes (Threshold > {config['attribute_threshold']}) ##")
    if len(predicted_attr_indices) > 0:
        for idx in predicted_attr_indices:
            print(f"  - {attribute_names[idx]} (Confidence: {attr_probs[idx]:.2f})")
    else:
        print("  - No attributes predicted above the threshold.")
        
    print("\n" + "-"*50)

if __name__ == '__main__':
    main()
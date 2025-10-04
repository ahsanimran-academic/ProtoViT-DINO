# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import our new modules
from data import SSLAttributeDataset, SSLAugmentation
from protovit_dino_model import ProtoViTBase, ProtoViT_DINO
from losses import DINOLoss, AttributeLoss, CoherenceDecorrelationLoss
from model import construct_PPNet # Your original function to get the ViT backbone

# --- Configuration ---
config = {
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 384, 4), # M, d, K
    'radius': 1,
    'sig_temp': 100.0,
    
    # Data paths
    'train_dir': './datasets/cub200_cropped/train_cropped/',
    'attributes_json': './datasets/cub_attributes.json',
    
    # Training parameters
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 0.04,
    'teacher_momentum': 0.996,
    'dino_out_dim': 4096, # DINO projection head output dimension
    
    # Loss weights
    'lambda_dino': 1.0,
    'lambda_attr': 0.5,
    'lambda_coh': 5e-3,
    'lambda_decorr': 1e-3,

    'save_dir': './saved_models/protovit_dino/',
}

# --- Main Training Logic ---
def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"Using device: {device}")

    # 1. Data
    ssl_transform = SSLAugmentation(img_size=config['img_size'])
    train_dataset = SSLAttributeDataset(
        root=config['train_dir'],
        attributes_json_path=config['attributes_json'],
        transform=ssl_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # 2. Model
    # Function to create the base model
    def create_base_model():
        # Use your original function to load the pretrained backbone
        vit_features = construct_PPNet(
            base_architecture=config['base_architecture'],
            pretrained=True,
            img_size=config['img_size'],
            prototype_shape=config['prototype_shape'],
            num_classes=1 # Dummy value
        ).features
        
        return ProtoViTBase(
            features=vit_features,
            img_size=config['img_size'],
            prototype_shape=config['prototype_shape'],
            radius=config['radius'],
            sig_temp=config['sig_temp']
        )

    model = ProtoViT_DINO(
        base_model_fn=create_base_model,
        out_dim=config['dino_out_dim'],
        teacher_momentum=config['teacher_momentum']
    ).to(device)
    
    # 3. Losses
    dino_loss_fn = DINOLoss(out_dim=config['dino_out_dim']).to(device)
    attr_loss_fn = AttributeLoss().to(device)
    coh_decorr_loss_fn = CoherenceDecorrelationLoss().to(device)
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # --- Training Loop ---
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for view1, view2, attributes, _ in train_loader:
            view1, view2, attributes = view1.to(device), view2.to(device), attributes.to(device)
            
            # Forward pass
            outputs = model(view1, view2)
            
            # Calculate losses
            # DINO loss (applied to both views symmetrically)
            loss_dino = dino_loss_fn(outputs['student_dino'][0], outputs['teacher_dino'][1]) + \
                        dino_loss_fn(outputs['student_dino'][1], outputs['teacher_dino'][0])
            loss_dino = loss_dino / 2.0
            
            # Attribute prediction loss
            loss_attr = attr_loss_fn(outputs['attr_preds'], attributes)
            
            # Coherence and Decorrelation loss on student prototypes
            student_protos = model.student.prototype_vectors
            student_slots = torch.sigmoid(model.student.patch_select * model.student.temp).squeeze(0)
            loss_coh, loss_decorr = coh_decorr_loss_fn(student_protos, student_slots)
            
            # Total weighted loss
            loss = (config['lambda_dino'] * loss_dino +
                    config['lambda_attr'] * loss_attr +
                    config['lambda_coh'] * loss_coh +
                    config['lambda_decorr'] * loss_decorr)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}, Dino: {loss_dino.item():.4f}, Attr: {loss_attr.item():.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.student.state_dict(), os.path.join(config['save_dir'], f'student_epoch_{epoch+1}.pth'))
            print(f"Saved student model checkpoint at epoch {epoch+1}")

if __name__ == '__main__':
    main()
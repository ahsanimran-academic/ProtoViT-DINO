# train.py (Final Corrected Version)
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import clip
from tqdm import tqdm

# Import our existing modules
from data import SSLAttributeDataset, SSLAugmentation
from protovit_dino_model import ProtoViTBase, ProtoViT_DINO
from losses import DINOLoss, AttributeLoss, CoherenceDecorrelationLoss
from model import construct_PPNet

# --- Configuration ---
config = {
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 768, 4),
    'radius': 1,
    'sig_temp': 100.0,

    'train_dir': './datasets/cub200_cropped/train_cropped_augmented/',
    'cub_root_dir': './datasets/CUB_200_2011/',
    'attributes_json': './datasets/cub_attributes.json',
    
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 5e-5,
    'weight_decay': 0.04,
    'teacher_momentum': 0.996,
    'dino_out_dim': 4096,
    'clip_out_dim': 512, # For the new head
    
    'lambda_dino': 1.0,
    'lambda_attr': 0.5,
    'lambda_clip': 0.2,
    'lambda_coh': 5e-3,
    'lambda_decorr': 1e-3,

    'save_dir': './saved_models/protovit_dino/',
    'clip_model_name': 'ViT-B/32',
}

def get_attribute_text_embeddings(cub_root_path, clip_model, device):
    attributes_file = os.path.join(cub_root_path, 'attributes', 'attributes.txt')
    if not os.path.exists(attributes_file):
        raise FileNotFoundError(f"Could not find attributes.txt at {attributes_file}")
    with open(attributes_file, 'r') as f: lines = f.readlines()
    concepts = [line.strip().split(' ', 1)[1].replace('has_', '').replace('::', ' ').replace('_', ' ') for line in lines]
    text_prompts = [f"a photo of a bird with {c}" for c in concepts]
    print(f"Encoding {len(concepts)} attribute text prompts with CLIP...")
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_embeddings = clip_model.encode_text(text_tokens)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"Using device: {device}")

    # Data
    ssl_transform = SSLAugmentation(img_size=config['img_size'])
    train_dataset = SSLAttributeDataset(root=config['train_dir'], attributes_json_path=config['attributes_json'], transform=ssl_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # Model
    def create_base_model():
        vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
        return ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape'], radius=config['radius'], sig_temp=config['sig_temp'])

    model = ProtoViT_DINO(
        base_model_fn=create_base_model,
        dino_out_dim=config['dino_out_dim'],
        clip_out_dim=config['clip_out_dim'],
        teacher_momentum=config['teacher_momentum']
    ).to(device)
    
    clip_model, _ = clip.load(config['clip_model_name'], device=device)
    for p in clip_model.parameters(): p.requires_grad = False
    attribute_embeddings = get_attribute_text_embeddings(config['cub_root_dir'], clip_model, device)
    
    # Losses & Optimizer
    dino_loss_fn = DINOLoss(out_dim=config['dino_out_dim']).to(device)
    attr_loss_fn = AttributeLoss().to(device)
    coh_decorr_loss_fn = CoherenceDecorrelationLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Training Loop
    for epoch in range(config['epochs']):
        model.train()
        loss_trackers = {'total': 0, 'dino': 0, 'attr': 0, 'clip': 0}
        
        for view1, view2, attributes, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            view1, view2, attributes = view1.to(device), view2.to(device), attributes.to(device)
            
            outputs = model(view1, view2)
            
            # DINO loss
            loss_dino = (dino_loss_fn(outputs['student_dino'][0], outputs['teacher_dino'][1]) + dino_loss_fn(outputs['student_dino'][1], outputs['teacher_dino'][0])) / 2.0
            
            # Attribute prediction loss
            loss_attr = attr_loss_fn(outputs['attr_preds'], attributes)
            
            # <<< FIX: Use the 'student_clip' output from the new head >>>
            num_attrs = attributes.sum(dim=1, keepdim=True).clamp(min=1)
            target_text_embeds = F.normalize((attributes @ attribute_embeddings.float()) / num_attrs, p=2, dim=1)
            
            z_clip_1 = F.normalize(outputs['student_clip'][0], p=2, dim=1)
            z_clip_2 = F.normalize(outputs['student_clip'][1], p=2, dim=1)
            
            loss_clip_1 = 1 - F.cosine_similarity(z_clip_1, target_text_embeds.float()).mean()
            loss_clip_2 = 1 - F.cosine_similarity(z_clip_2, target_text_embeds.float()).mean()
            loss_clip = (loss_clip_1 + loss_clip_2) / 2.0

            # Structural Priors
            student_protos = model.student.prototype_vectors
            student_slots = torch.sigmoid(model.student.patch_select * model.student.temp).squeeze(0)
            loss_coh, loss_decorr = coh_decorr_loss_fn(student_protos, student_slots)
            
            # Total weighted loss
            loss = (config['lambda_dino'] * loss_dino +
                    config['lambda_attr'] * loss_attr +
                    config['lambda_clip'] * loss_clip +
                    config['lambda_coh'] * loss_coh +
                    config['lambda_decorr'] * loss_decorr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_trackers['total'] += loss.item()
            loss_trackers['dino'] += loss_dino.item()
            loss_trackers['attr'] += loss_attr.item()
            loss_trackers['clip'] += loss_clip.item()
            
        avg_loss = {k: v / len(train_loader) for k, v in loss_trackers.items()}
        print(f"Epoch [{epoch+1}/{config['epochs']}], Avg Loss: {avg_loss['total']:.4f}, Dino: {avg_loss['dino']:.4f}, Attr: {avg_loss['attr']:.4f}, CLIP: {avg_loss['clip']:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.student.state_dict(), os.path.join(config['save_dir'], f'student_epoch_{epoch+1}.pth'))
            print(f"Saved student model checkpoint at epoch {epoch+1}")

if __name__ == '__main__':
    main()
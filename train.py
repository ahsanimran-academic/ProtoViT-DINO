# train.py (Fully Corrected Final Version)
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import clip
from tqdm import tqdm
import math

# Import our existing modules
from data import SSLAttributeDataset, SSLAugmentation
from protovit_dino_model import ProtoViTBase, ProtoViT_DINO
from losses import DINOLoss, AttributeLoss, CoherenceDecorrelationLoss, PrototypeNCELoss
from model import construct_PPNet

# --- Configuration ---
# Using DeiT-Small as in your provided code
config = {
    'base_architecture': 'deit_base_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 768, 4), # Dimension must be 768
    'radius': 1,
    'sig_temp': 100.0,

    'train_dir': './datasets/cub200_cropped/train_cropped',
    'cub_root_dir': './datasets/CUB_200_2011/',
    'attributes_json': './datasets/cub_attributes.json',
    
    'epochs': 50,
    'batch_size': 128,
    'learning_rate': 5e-5,
    'weight_decay': 0.04,
    'teacher_momentum': 0.996,
    'dino_out_dim': 4096,
    
    'lambda_dino': 1.0,
    'lambda_attr': 0.5,
    'lambda_clip': 0.3,
    'lambda_proto_nce': 0.5,
    'lambda_coh': 5e-4,
    'lambda_decorr': 1e-6,

    'save_dir': './saved_models/protovit_dino/',
    'clip_model_name': 'ViT-B/32',
}

# <<< NEW: A simple logger to write to a file and console >>>
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, 'w') as f: f.write('')
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f: f.write(message + '\n')

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
    logger = Logger(os.path.join(config['save_dir'], 'train_log.txt'))
    logger.log(f"Using device: {device}")
    logger.log("Configuration:\n" + str(config))

    # Data
    ssl_transform = SSLAugmentation(img_size=config['img_size'])
    train_dataset = SSLAttributeDataset(root=config['train_dir'], attributes_json_path=config['attributes_json'], transform=ssl_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # Model
    def create_base_model():
        vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
        return ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape'], radius=config['radius'], sig_temp=config['sig_temp'])

    # 1. Load the frozen CLIP model first
    clip_model, _ = clip.load(config['clip_model_name'], device=device)
    for p in clip_model.parameters(): p.requires_grad = False
    attribute_embeddings = get_attribute_text_embeddings(config['cub_root_dir'], clip_model, device)
    
    # 2. Pass the frozen clip_model into the ProtoViT_DINO constructor
    model = ProtoViT_DINO(
        base_model_fn=create_base_model,
        clip_model=clip_model, # Pass the loaded model here
        dino_out_dim=config['dino_out_dim'],
        teacher_momentum=config['teacher_momentum']
    ).to(device)
    
    
    # Losses & Optimizer
    dino_loss_fn = DINOLoss(out_dim=config['dino_out_dim']).to(device)
    attr_loss_fn = AttributeLoss().to(device)
    coh_decorr_loss_fn = CoherenceDecorrelationLoss().to(device)
    proto_nce_loss_fn = PrototypeNCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best_loss = float('inf')

    # Training Loop
    for epoch in range(config['epochs']):
        model.train()
        loss_trackers = {'total': 0, 'dino': 0, 'attr': 0, 'clip': 0, 'proto_nce': 0, 'coh': 0, 'decorr': 0}
        
        for view1, view2, attributes, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            view1, view2, attributes = view1.to(device), view2.to(device), attributes.to(device)
            
            outputs = model(view1, view2)
            
            # --- FIX: Using the "One-vs-Many" loss calculation ---
            loss_dino = (dino_loss_fn(outputs['student_dino'][0], outputs['teacher_dino'][1]) + dino_loss_fn(outputs['student_dino'][1], outputs['teacher_dino'][0])) / 2.0
            loss_attr = attr_loss_fn(outputs['attr_preds'], attributes)
            s1, s2 = outputs['student_protos'][0], outputs['student_protos'][1]
            loss_proto_nce = proto_nce_loss_fn(s1, s2)
            student_protos, student_slots = model.student.prototype_vectors, torch.sigmoid(model.student.patch_select * model.student.temp).squeeze(0)
            loss_coh, loss_decorr = coh_decorr_loss_fn(student_protos, student_slots)
            
            z_clip_1, z_clip_2 = F.normalize(outputs['student_clip'][0]), F.normalize(outputs['student_clip'][1])
            batch_losses_clip = []
            for i in range(z_clip_1.shape[0]):
                true_attr_indices = attributes[i].nonzero(as_tuple=True)[0]
                if len(true_attr_indices) == 0: continue
                target_text_embeds = attribute_embeddings[true_attr_indices].float()
                sims_1 = F.cosine_similarity(z_clip_1[i].unsqueeze(0), target_text_embeds)
                sims_2 = F.cosine_similarity(z_clip_2[i].unsqueeze(0), target_text_embeds)
                loss_for_item = (1 - sims_1.mean()) + (1 - sims_2.mean())
                batch_losses_clip.append(loss_for_item)

            if batch_losses_clip:
                loss_clip = torch.stack(batch_losses_clip).mean() / 2.0
            else:
                loss_clip = torch.tensor(0.0).to(device)

            # Total weighted loss
            loss = (config['lambda_dino'] * loss_dino +
                    config['lambda_attr'] * loss_attr +
                    config['lambda_clip'] * loss_clip +
                    config['lambda_proto_nce'] * loss_proto_nce +
                    config['lambda_coh'] * loss_coh +
                    config['lambda_decorr'] * loss_decorr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for k, v in [('total', loss), ('dino', loss_dino), ('attr', loss_attr), ('clip', loss_clip), ('proto_nce', loss_proto_nce), ('coh', loss_coh), ('decorr', loss_decorr)]:
                loss_trackers[k] += v.item()
            
        avg_losses = {k: v / len(train_loader) for k, v in loss_trackers.items()}
        log_message = f"Epoch [{epoch+1}/{config['epochs']}], Avg Loss: {avg_losses['total']:.4f} | " + \
                      f"Dino: {avg_losses['dino']:.4f} | Attr: {avg_losses['attr']:.4f} | CLIP: {avg_losses['clip']:.4f} | " + \
                      f"ProtoNCE: {avg_losses['proto_nce']:.4f} | Coh: {avg_losses['coh']:.4f} | Decorr: {avg_losses['decorr']:.4f}"
        logger.log(log_message)

        current_loss = avg_losses['total']
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.student.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            logger.log(f"*** New best model saved with loss: {best_loss:.4f} ***")

if __name__ == '__main__':
    main()
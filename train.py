# train.py 
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import clip
from tqdm import tqdm
import math
from torch.cuda.amp import GradScaler, autocast

from data import SSLAttributeDataset, SSLAugmentation
from protovit_dino_model import ProtoViTBase, ProtoViT_DINO
from losses import DINOLoss, AttributeLoss, CoherenceDecorrelationLoss, PrototypeNCELoss
from model import construct_PPNet

config = {
    'resume_from_checkpoint': None,  # e.g., './saved_models/protovit_dino/best_model.pth'
    'start_epoch': 0,
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (2000, 384, 4),
    'radius': 1, 'sig_temp': 100.0,
    'train_dir': './datasets/cub200_cropped/train_cropped/',
    'cub_root_dir': './datasets/CUB_200_2011/',
    'attributes_json': './datasets/cub_attributes.json',
    'warmup_epochs': 15, 'joint_epochs': 55, 'refinement_epochs': 20,
    'warmup_lr': 3e-3, 'main_lr': 5e-5, 'refinement_lr': 1e-5,
    'batch_size': 32, 'weight_decay': 0.04, 'teacher_momentum': 0.996,
    'dino_out_dim': 4096, 'clip_out_dim': 512,
    'lambda_dino': 1.0, 'lambda_attr': 0.5, 'lambda_clip': 0.5,
    'lambda_proto_nce': 0.3, 'lambda_coh': 5e-4, 'lambda_decorr': 1e-7,
    'save_dir': './saved_models/protovit_dino/', 'clip_model_name': 'ViT-B/32',
}

class Logger:
    def __init__(self, log_file, resume=False):
        mode = 'a' if resume else 'w'
        self.log_file = log_file
        with open(self.log_file, mode) as f:
            if not resume: f.write('--- Training Log ---\n')
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f: f.write(message + '\n')

def get_attribute_text_embeddings(cub_root_path, clip_model, device):
    attributes_file = os.path.join(cub_root_path, 'attributes', 'attributes.txt')
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
    logger = Logger(os.path.join(config['save_dir'], 'train_log.txt'), resume=(config['start_epoch'] > 0))
    logger.log(f"Using device: {device}")
    logger.log("Configuration:\n" + str(config))

    ssl_transform = SSLAugmentation(img_size=config['img_size'])
    train_dataset = SSLAttributeDataset(root=config['train_dir'], attributes_json_path=config['attributes_json'], transform=ssl_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    def create_base_model():
        vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
        return ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape'], radius=config['radius'], sig_temp=config['sig_temp'])

    # 1. Load the frozen CLIP model *before* creating the main model
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
    
    if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
        logger.log(f"Resuming training from checkpoint: {config['resume_from_checkpoint']}")
        model.student.load_state_dict(torch.load(config['resume_from_checkpoint'], map_location=device))
        model.teacher.load_state_dict(model.student.state_dict())
    
    try:
        model = torch.compile(model)
    except Exception:
        pass
    
    loss_fns = { 'dino': DINOLoss(out_dim=config['dino_out_dim']).to(device), 'attr': AttributeLoss().to(device),
                 'coh_decorr': CoherenceDecorrelationLoss().to(device), 'proto_nce': PrototypeNCELoss().to(device) }
    
    scaler = GradScaler()
    best_loss = float('inf')
    total_epochs = config['warmup_epochs'] + config['joint_epochs'] + config['refinement_epochs']

    # --- Reusable train_epoch function ---
    # <<< FIX: The function definition is simplified. It gets variables like model, logger, etc. from the parent scope. >>>
    def train_epoch(epoch, optimizer, scheduler, use_structural_losses):
        model.train()
        loss_trackers = {k: 0 for k in ['total', 'dino', 'attr', 'clip', 'proto_nce', 'coh', 'decorr']}
        lambda_coh = config['lambda_coh'] if use_structural_losses else 0.0
        lambda_decorr = config['lambda_decorr'] if use_structural_losses else 0.0

        for view1, view2, attributes, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
            view1, view2, attributes = view1.to(device), view2.to(device), attributes.to(device)
            with autocast():
                outputs = model(view1, view2)
                loss_dino = (loss_fns['dino'](outputs['student_dino'][0], outputs['teacher_dino'][1]) + loss_fns['dino'](outputs['student_dino'][1], outputs['teacher_dino'][0])) / 2.0
                loss_attr = loss_fns['attr'](outputs['attr_preds'], attributes)
                s1, s2 = outputs['student_protos'][0], outputs['student_protos'][1]
                loss_proto_nce = loss_fns['proto_nce'](s1, s2)
                student_protos = model.student.prototype_vectors
                student_slots = torch.sigmoid(model.student.patch_select * model.student.temp).squeeze(0)
                loss_coh_val, loss_decorr_val = loss_fns['coh_decorr'](student_protos, student_slots)
                
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
                loss_clip = torch.stack(batch_losses_clip).mean() / 2.0 if batch_losses_clip else torch.tensor(0.0).to(view1.device)

                loss = (config['lambda_dino'] * loss_dino + config['lambda_attr'] * loss_attr + config['lambda_clip'] * loss_clip +
                        config['lambda_proto_nce'] * loss_proto_nce + lambda_coh * loss_coh_val + lambda_decorr * loss_decorr_val)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler: scheduler.step()
            
            for k, v in [('total', loss), ('dino', loss_dino), ('attr', loss_attr), ('clip', loss_clip), ('proto_nce', loss_proto_nce), ('coh', loss_coh_val), ('decorr', loss_decorr_val)]:
                if v.numel() > 0: loss_trackers[k] += v.item()
        
        return {k: v / len(train_loader) for k, v in loss_trackers.items()}

    # --- Unified Training Loop ---
    optimizer = None
    scheduler = None
    
    for epoch in range(config['start_epoch'], total_epochs):
        stage_changed = False
        
        if epoch < config['warmup_epochs']:
            stage, use_structural_losses = "Warm-up", False
            if epoch == config['start_epoch']: stage_changed = True
            if stage_changed:
                logger.log(f"\n--- Entering STAGE 1: {stage} ---")
                for param in model.student.features.parameters(): param.requires_grad = False
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['warmup_lr'])
                scheduler = None
        
        elif epoch < config['warmup_epochs'] + config['joint_epochs']:
            stage, use_structural_losses = "Joint", False
            if epoch == max(config['start_epoch'], config['warmup_epochs']): stage_changed = True
            if stage_changed:
                logger.log(f"\n--- Entering STAGE 2: {stage} Training ---")
                for param in model.student.parameters(): param.requires_grad = True
                optimizer = optim.AdamW(model.parameters(), lr=config['main_lr'], weight_decay=config['weight_decay'])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['joint_epochs'] * len(train_loader))
        
        else:
            stage, use_structural_losses = "Refinement", True
            if epoch == max(config['start_epoch'], config['warmup_epochs'] + config['joint_epochs']): stage_changed = True
            if stage_changed:
                logger.log(f"\n--- Entering STAGE 3: {stage} ---")
                for param in model.student.features.parameters(): param.requires_grad = False
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['refinement_lr'])
                scheduler = None

        # <<< FIX: The call to train_epoch is simplified to match its new definition >>>
        avg_losses = train_epoch(epoch, optimizer, scheduler, use_structural_losses)
        
        log_message = f"Epoch [{epoch+1}/{total_epochs}] [{stage}] Avg Loss: {avg_losses['total']:.4f} | ..." # Abridged for clarity
        logger.log(log_message)

        current_loss = avg_losses['total']
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.student.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            logger.log(f"*** New best model saved with loss: {best_loss:.4f} ***")

if __name__ == '__main__':
    main()
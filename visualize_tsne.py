# visualize_tsne.py (Corrected)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/student_epoch_10.pth',
    'output_file': './tsne_prototypes.png',
    
    # --- MODEL ARCHITECTURE (Must match the trained model) ---
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 384, 4), # Make sure this matches your trained model
    
    # --- t-SNE PARAMETERS ---
    'perplexity': 30,
    'max_iter': 1000, # <<< FIX: Renamed from n_iter to max_iter
    'learning_rate': 'auto',
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. ## LOAD THE TRAINED STUDENT MODEL ##
    print("Loading student model...")
    vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
    model = ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape']).to(device)
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()

    # 2. ## EXTRACT AND AVERAGE PROTOTYPES ##
    with torch.no_grad():
        proto_vectors = model.prototype_vectors.detach()
        avg_proto_vectors = proto_vectors.mean(dim=-1).cpu().numpy()
    
    print(f"Extracted {avg_proto_vectors.shape[0]} averaged prototype vectors.")

    # 3. ## RUN t-SNE ##
    print("Running t-SNE... (This may take a minute)")
    tsne = TSNE(
        n_components=2,
        perplexity=config['perplexity'],
        max_iter=config['max_iter'], # <<< FIX: Using the correct parameter name
        init='pca',
        learning_rate=config['learning_rate'],
        random_state=42
    )
    tsne_results = tsne.fit_transform(avg_proto_vectors)

    # 4. ## PLOT THE RESULTS ##
    print("Plotting results...")
    plt.figure(figsize=(12, 12))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6)
    
    for i in range(0, tsne_results.shape[0], 10):
        plt.text(tsne_results[i, 0], tsne_results[i, 1], str(i), fontsize=8)

    plt.title('t-SNE Visualization of 256 Learned Prototypes')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    
    plt.savefig(config['output_file'])
    print(f"t-SNE plot saved to {config['output_file']}")
    plt.close()

if __name__ == '__main__':
    main()
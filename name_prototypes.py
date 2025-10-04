# name_prototypes.py (Final Version with Corrected PCA)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import clip
from tqdm import tqdm
from sklearn.decomposition import PCA

# --- Import from your existing model files ---
from model import construct_PPNet
from protovit_dino_model import ProtoViTBase

# --- Main Configuration ---
config = {
    # --- PATHS ---
    'student_checkpoint_path': './saved_models/protovit_dino/student_epoch_100.pth',
    'cub_root_dir': './datasets/CUB_200_2011/',
    'projection_dir': './prototype_projections/',
    'output_file': './prototype_analysis_v3.html',
    
    # --- MODEL ARCHITECTURE ---
    'base_architecture': 'deit_small_patch16_224',
    'img_size': 224,
    'prototype_shape': (256, 384, 4),
    
    # --- NAMING PARAMETERS ---
    'top_k_names': 5,
    'clip_model_name': 'ViT-B/32'
}

# --- HELPER FUNCTIONS ---

def generate_concept_list(cub_root_path):
    attributes_file = os.path.join(cub_root_path, 'attributes', 'attributes.txt')
    if not os.path.exists(attributes_file):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find 'attributes.txt' at the expected path: {attributes_file}")
    with open(attributes_file, 'r') as f:
        lines = f.readlines()
    concepts = [line.strip().split(' ', 1)[1].replace('has_', '').replace('::', ' ').replace('_', ' ') for line in lines]
    return concepts

def create_html_visualization(output_path, proto_info, M, K):
    html = "<html><head><title>Prototype Analysis</title><style>"
    html += "body { font-family: sans-serif; } table { border-collapse: collapse; width: 100%; } "
    html += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; } "
    html += "th { background-color: #f2f2f2; } img { width: 64px; height: 64px; object-fit: cover; } "
    html += "td:first-child { width: 5%; } "
    html += "ul { margin: 0; padding-left: 20px; } "
    html += "</style></head><body><h1>Prototype Visualization</h1><table>"
    html += "<tr><th>Prototype ID</th>"
    for k in range(K):
        html += f"<th>Sub-prototype {k}</th>"
    html += "<th>Top Concept Names (Averaged)</th></tr>"
    for j in range(M):
        html += f"<tr><td><b>{j}</b></td>"
        for k in range(K):
            img_path = proto_info[j]['sub_prototypes'][k]['img_path']
            relative_img_path = os.path.relpath(img_path, os.path.dirname(output_path))
            html += f"<td><img src='{relative_img_path}' alt='p{j}k{k}'></td>"
        names_html = "<ul>"
        for name, score in proto_info[j]['names']:
            names_html += f"<li>{name} ({score:.3f})</li>"
        names_html += "</ul>"
        html += f"<td>{names_html}</td></tr>"
    html += "</table></body></html>"
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Visualization saved to {output_path}")

# <<< DEFINITIVE FIX: Rewritten get_pca_projection function >>>
def get_pca_projection(prototypes, target_dim):
    """
    Computes a PCA-based projection matrix of shape (target_dim, source_dim).
    This version is simpler and more robust.
    """
    M, D, K = prototypes.shape # D is the source dimension, e.g., 384
    proto_flat = prototypes.permute(0, 2, 1).reshape(M * K, D).detach().cpu().numpy()
    
    # The number of PCA components can't be more than the number of features.
    n_components = min(proto_flat.shape[0], D)
    
    pca = PCA(n_components=n_components)
    pca.fit(proto_flat)
    
    # This matrix has shape (n_components, D), e.g., (384, 384)
    pca_matrix = torch.from_numpy(pca.components_).float()
    
    # Create the final projection matrix of the correct size, initialized to zeros
    # This guarantees the shape is (target_dim, D), e.g., (512, 384)
    projection_matrix = torch.zeros(target_dim, D)
    
    # Copy the PCA components into the top part of the matrix.
    # The remaining rows will be zero, which is a valid projection.
    projection_matrix[:n_components, :] = pca_matrix
    
    return projection_matrix.to(prototypes.device)


# --- MAIN EXECUTION ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. ## LOAD MODEL AND CLIP ##
    print("Loading models...")
    vit_features = construct_PPNet(base_architecture=config['base_architecture'], pretrained=True, img_size=config['img_size'], prototype_shape=config['prototype_shape'], num_classes=1).features
    model = ProtoViTBase(features=vit_features, img_size=config['img_size'], prototype_shape=config['prototype_shape']).to(device)
    model.load_state_dict(torch.load(config['student_checkpoint_path'], map_location=device))
    model.eval()
    clip_model, _ = clip.load(config['clip_model_name'], device=device)
    print("Models loaded.")

    # 2. ## GENERATE AND ENCODE TEXT CONCEPTS ##
    print("Generating and encoding text concepts...")
    concepts = generate_concept_list(config['cub_root_dir'])
    text_prompts = [f"a photo of a bird with {c}" for c in concepts]
    text_tokens = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 3. ## GET PROTOTYPE FEATURES AND COMPUTE SIMILARITY ##
    print("Computing similarity between prototypes and text concepts...")
    with torch.no_grad():
        proto_vectors = model.prototype_vectors.detach()
        avg_proto_vectors = proto_vectors.mean(dim=-1)
        avg_proto_vectors /= avg_proto_vectors.norm(dim=-1, keepdim=True)

        proto_dim = avg_proto_vectors.shape[1]
        clip_dim = text_features.shape[1]

        if proto_dim != clip_dim:
            print("Creating PCA-based projection...")
            projection_matrix = get_pca_projection(proto_vectors, clip_dim)
            print(f"Projection matrix shape: {projection_matrix.shape}") # Should now be [512, 384]
            
            avg_proto_vectors = F.linear(avg_proto_vectors, projection_matrix)
            print(f"Projected prototype vectors shape: {avg_proto_vectors.shape}") # Should now be [256, 512]
            
            avg_proto_vectors /= avg_proto_vectors.norm(dim=-1, keepdim=True)
            
        similarity = avg_proto_vectors @ text_features.float().T
    
    # 4. ## FIND TOP NAMES AND CREATE VISUALIZATION ##
    print("Finding top names and preparing visualization...")
    top_k_scores, top_k_indices = torch.topk(similarity, config['top_k_names'], dim=-1)

    M, _, K = config['prototype_shape']
    prototype_info = {}

    for j in range(M):
        prototype_info[j] = {'sub_prototypes': [], 'names': []}
        for k in range(K):
            prototype_info[j]['sub_prototypes'].append({
                'img_path': os.path.join(config['projection_dir'], f'prototype-j{j}-k{k}.png')
            })
        for i in range(config['top_k_names']):
            concept_name = concepts[top_k_indices[j, i].item()]
            score = top_k_scores[j, i].item()
            prototype_info[j]['names'].append((concept_name, score))

    create_html_visualization(config['output_file'], prototype_info, M, K)

if __name__ == '__main__':
    main()
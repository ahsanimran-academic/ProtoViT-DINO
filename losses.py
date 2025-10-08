# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    """
    DINO loss function.
    Includes teacher-student cross-entropy, centering, and sharpening.
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        
        # Teacher sharpening and centering
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        
        # Cross-entropy loss
        # Each view from the student tries to predict the teacher's output for other views
        loss = -torch.sum(teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class AttributeLoss(nn.Module):
    """
    Simple wrapper for Binary Cross-Entropy to predict attributes.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, predicted_attrs, true_attrs):
        return self.loss_fn(predicted_attrs, true_attrs)

class CoherenceDecorrelationLoss(nn.Module):
    """
    Combines prototype coherence with a decorrelation regularizer.
    """
    def forward(self, prototype_vectors, slots):
        # 1. Coherence Loss (from your original model)
        proto_vecs_norm = F.normalize(prototype_vectors, p=2, dim=1)
        coherence_losses = []
        for j in range(prototype_vectors.shape[0]):
            active_mask = slots[j] > 0.5
            if active_mask.sum() > 1:
                active_protos = proto_vecs_norm[j, :, active_mask]
                mean_proto = active_protos.mean(dim=1, keepdim=True)
                coh_loss = 1 - F.cosine_similarity(active_protos, mean_proto, dim=0).mean()
                coherence_losses.append(coh_loss)
        
        loss_coh = torch.stack(coherence_losses).mean() if coherence_losses else torch.tensor(0.0).to(prototype_vectors.device)
        
        # 2. Decorrelation Loss (replaces orthogonality)
        proto_flat = proto_vecs_norm.permute(0, 2, 1).reshape(prototype_vectors.shape[0] * prototype_vectors.shape[2], -1)
        proto_centered = proto_flat - proto_flat.mean(dim=0, keepdim=True)
        cov_matrix = (proto_centered.T @ proto_centered) / (proto_centered.shape[0] - 1)
        loss_decorr = (cov_matrix - torch.eye(cov_matrix.shape[0]).to(cov_matrix.device)).pow(2).sum()

        return loss_coh, loss_decorr

class PrototypeNCELoss(nn.Module):
    """
    InfoNCE-style contrastive loss for prototype score vectors.
    Pushes prototype activations from two views of the same image together,
    and pushes them apart from activations of other images in the batch.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, scores1, scores2):
        # scores1 and scores2 shape: [batch_size, num_prototypes]
        
        # L2 normalize the feature vectors
        scores1 = F.normalize(scores1, dim=1)
        scores2 = F.normalize(scores2, dim=1)
        
        # Calculate cosine similarity matrix
        # The (i, j) entry is the similarity between view1 of image i and view2 of image j
        logits = torch.mm(scores1, scores2.T) / self.temperature
        
        # The positive pairs are on the diagonal (i, i)
        # Create ground-truth labels
        batch_size = scores1.shape[0]
        labels = torch.arange(batch_size, device=scores1.device)
        
        # Calculate loss symmetrically
        loss_12 = self.loss_fn(logits, labels)
        loss_21 = self.loss_fn(logits.T, labels)
        
        return (loss_12 + loss_21) / 2.0
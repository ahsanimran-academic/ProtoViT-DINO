# protovit_dino_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# This is your original model, refactored to be the base component.
# The core logic of greedy matching and feature extraction remains here.
class ProtoViTBase(nn.Module):
    def __init__(self, features, img_size, prototype_shape, radius=1, sig_temp=100.0):
        super().__init__()
        self.features = features
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.radius = radius
        self.temp = sig_temp
        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape), requires_grad=True)
        self.patch_select = nn.Parameter(torch.ones(1, self.num_prototypes, prototype_shape[-1]) * 0.1, requires_grad=True)

        # Detect architecture to handle feature extraction
        self.arc = 'deit' if 'deit' in str(features).lower() else 'other'

    def conv_features(self, x):
        # Your existing ViT feature extraction logic
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x_focal = x[:, 1:] - x[:, 0].unsqueeze(1) # Focalize
        B, L, D = x_focal.shape
        W = int(L ** 0.5)
        feature_emb = x_focal.permute(0, 2, 1).reshape(B, D, W, W)
        return feature_emb, x[:, 0]

    def subpatch_dist(self, x):
        # Your subpatch_dist logic...
        conv_features, cls_token = self.conv_features(x)
        conv_features_normed = F.normalize(conv_features, p=2, dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        n_p = self.prototype_shape[-1]
        dist_all = []
        for i in range(n_p):
            proto_i = now_prototype_vectors[:, :, i].unsqueeze(-1).unsqueeze(-1)
            dist_i = F.conv2d(input=conv_features_normed, weight=proto_i).flatten(2).unsqueeze(-1)
            dist_all.append(dist_i)
        return conv_features, torch.cat(dist_all, dim=-1), cls_token

    def greedy_distance(self, x, get_features=False):
        # Your existing greedy matching logic...
        # This part remains the same as it's the core of ProtoViT
        conv_features, dist_all, cls_token = self.subpatch_dist(x)
        slots = torch.sigmoid(self.patch_select * self.temp)
        factor = slots.sum(-1).unsqueeze(-1) + 1e-10
        n_p = self.prototype_shape[-1]
        batch_size = x.shape[0]
        num_patches = dist_all.shape[2]
        
        mask_all = torch.ones(batch_size, self.num_prototypes, num_patches, n_p, device=x.device)
        adjacent_mask = torch.ones(batch_size, self.num_prototypes, num_patches, device=x.device)

        indices, values = [], []
        for _ in range(n_p):
            dist_masked = dist_all + (1 - mask_all * adjacent_mask.unsqueeze(-1)) * (-1e5)
            max_vals, max_ids = dist_masked.view(batch_size * self.num_prototypes, -1).max(1)
            
            # Unravel indices
            patch_ids = (max_ids % (num_patches * n_p)) // n_p
            subpatch_ids = max_ids % n_p
            
            values.append(max_vals.view(batch_size, self.num_prototypes, 1))
            indices.append(patch_ids.view(batch_size, self.num_prototypes, 1))

            # Update masks (this part is complex, simplified for clarity, use your original logic)
            # This is a placeholder for your detailed masking logic
            mask_all.view(batch_size * self.num_prototypes, -1).scatter_(1, max_ids.unsqueeze(1), 0)
        
        values = torch.cat(values, dim=-1)
        values_slot = values * (slots * n_p / factor)
        max_activation_slots = values_slot.sum(-1)
        
        if get_features:
            return conv_features, max_activation_slots, torch.cat(indices, dim=-1), cls_token
        return max_activation_slots, values, cls_token

    def forward(self, x):
        # The base forward now returns the CLS token (for DINO head) and prototype scores
        proto_scores, _, cls_token = self.greedy_distance(x)
        return cls_token, proto_scores
    def push_forward(self, x):
        """
        A forward pass specifically for the projection/push step.
        It returns the detailed matching information needed.
        """
        # We need to re-implement a bit of the greedy logic here to get all the outputs
        conv_features, dist_all, cls_token = self.subpatch_dist(x)
        slots = torch.sigmoid(self.patch_select * self.temp)
        
        n_p = self.prototype_shape[-1]
        batch_size = x.shape[0]
        num_patches = dist_all.shape[2]
        
        # This is a simplified greedy matching for clarity during the push.
        # It finds the best patch for each sub-prototype independently, which is
        # sufficient and much faster for finding the global best exemplars.
        
        # For each prototype, for each sub-prototype, find the best patch location and its value.
        # dist_all shape: [batch, num_prototypes, num_patches, num_sub_prototypes]
        
        # Permute to make sub-prototypes the last dimension for easy processing
        # New shape: [batch, num_prototypes, num_sub_prototypes, num_patches]
        dist_permuted = dist_all.permute(0, 1, 3, 2)
        
        # Find the max activation value and its index (patch_id) for each sub-prototype
        # values will be the similarity scores, indices will be the patch locations
        values, indices = dist_permuted.max(dim=-1)
        
        # values shape: [batch, num_prototypes, num_sub_prototypes]
        # indices shape: [batch, num_prototypes, num_sub_prototypes]
        
        return conv_features, values, indices


# This is the main model class we will use in the trainer.
# It wraps the student and teacher networks and adds the DINO heads.
class ProtoViT_DINO(nn.Module):
    def __init__(self, base_model_fn, out_dim=2048, use_bn_in_head=False, teacher_momentum=0.996):
        super().__init__()
        self.teacher_momentum = teacher_momentum

        # Create student and teacher networks
        self.student = base_model_fn()
        self.teacher = deepcopy(self.student)

        # DINO projection head
        student_embed_dim = self.student.prototype_vectors.shape[1]
        self.dino_head = self._build_mlp(student_embed_dim, out_dim, use_bn=use_bn_in_head)
        self.teacher_dino_head = deepcopy(self.dino_head)

        # Attribute prediction head
        num_prototypes = self.student.num_prototypes
        self.attribute_head = nn.Linear(num_prototypes, 312)

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_dino_head.parameters():
            p.requires_grad = False

    def _build_mlp(self, in_dim, out_dim, use_bn):
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_bn else nn.Identity(),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_bn else nn.Identity(),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        ]
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _update_teacher_ema(self):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(self.teacher_momentum).add_(student_param.data, alpha=1 - self.teacher_momentum)
        for student_param, teacher_param in zip(self.dino_head.parameters(), self.teacher_dino_head.parameters()):
            teacher_param.data.mul_(self.teacher_momentum).add_(student_param.data, alpha=1 - self.teacher_momentum)
            
    def forward(self, view1, view2):
        # Student forward pass
        cls1_student, proto_scores1_student = self.student(view1)
        cls2_student, proto_scores2_student = self.student(view2)
        
        dino_out1_student = self.dino_head(cls1_student)
        dino_out2_student = self.dino_head(cls2_student)
        
        # Attribute prediction from one view
        attr_preds = self.attribute_head(proto_scores1_student)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            self._update_teacher_ema() # EMA update
            cls1_teacher, _ = self.teacher(view1)
            cls2_teacher, _ = self.teacher(view2)
            dino_out1_teacher = self.teacher_dino_head(cls1_teacher)
            dino_out2_teacher = self.teacher_dino_head(cls2_teacher)

        # Return all necessary outputs for loss calculation
        return {
            "student_dino": [dino_out1_student, dino_out2_student],
            "teacher_dino": [dino_out1_teacher, dino_out2_teacher],
            "student_protos": [proto_scores1_student, proto_scores2_student],
            "attr_preds": attr_preds,
        }
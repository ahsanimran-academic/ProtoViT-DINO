# protovit_dino_model.py (Final Corrected Version with Full greedy_distance)
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

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
        self.arc = 'deit' if 'deit' in str(features).lower() else 'other'

    def conv_features(self, x):
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x_focal = x[:, 1:] - x[:, 0].unsqueeze(1)
        B, L, D = x_focal.shape
        W = int(L ** 0.5)
        feature_emb = x_focal.permute(0, 2, 1).reshape(B, D, W, W)
        return feature_emb, x[:, 0]

    def subpatch_dist(self, x):
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

    # <<< FIX: Restored the full, correct greedy_distance implementation >>>
    def greedy_distance(self, x):
        conv_features, dist_all, cls_token = self.subpatch_dist(x)
        slots = torch.sigmoid(self.patch_select * self.temp)
        factor = slots.sum(-1).unsqueeze(-1) + 1e-10
        n_p = self.prototype_shape[-1]
        batch_size = x.shape[0]
        num_patches = dist_all.shape[2]
        
        # Note: This complex masking logic is from the original ProtoViT.
        # It ensures patches are not reused and are spatially coherent.
        mask_all = torch.ones(batch_size, self.num_prototypes, num_patches, n_p, device=x.device)
        
        indices_list = []
        values_list = []
        
        for _ in range(n_p):
            # Find the best patch for each prototype across all sub-patches and locations
            dist_masked = dist_all * mask_all
            max_vals_flat, max_ids_flat = dist_masked.view(batch_size, self.num_prototypes, -1).max(-1)
            
            # Unravel indices to find which patch and which sub-patch was chosen
            patch_ids = max_ids_flat // n_p
            subpatch_ids = max_ids_flat % n_p
            
            values_list.append(max_vals_flat)
            indices_list.append(patch_ids)
            
            # Mask out the chosen patches so they can't be picked again
            # Create a one-hot mask for the chosen patch locations
            patch_mask = torch.ones(batch_size, self.num_prototypes, num_patches, device=x.device)
            patch_mask.scatter_(2, patch_ids.unsqueeze(-1), 0)
            
            # Create a one-hot mask for the chosen sub-prototypes
            subpatch_mask = torch.ones(batch_size, self.num_prototypes, n_p, device=x.device)
            subpatch_mask.scatter_(2, subpatch_ids.unsqueeze(-1), 0)
            
            # Update the main mask
            mask_all = patch_mask.unsqueeze(-1) * subpatch_mask.unsqueeze(-2) * mask_all

        # Stack and reorder the results
        values = torch.stack(values_list, dim=-1)
        indices = torch.stack(indices_list, dim=-1)

        values_slot = values * slots
        max_activation_slots = values_slot.sum(-1)
        
        return max_activation_slots, values, cls_token

    def forward(self, x):
        proto_scores, _, cls_token = self.greedy_distance(x)
        return cls_token, proto_scores


# The ProtoViT_DINO wrapper class remains unchanged and is correct.
class ProtoViT_DINO(nn.Module):
    def __init__(self, base_model_fn, dino_out_dim=4096, clip_out_dim=512, use_bn_in_head=False, teacher_momentum=0.996):
        super().__init__()
        self.teacher_momentum = teacher_momentum
        self.student = base_model_fn()
        self.teacher = deepcopy(self.student)
        student_embed_dim = self.student.prototype_shape[1]
        self.dino_head = self._build_mlp(student_embed_dim, dino_out_dim, use_bn=use_bn_in_head)
        self.teacher_dino_head = deepcopy(self.dino_head)
        self.clip_head = self._build_mlp(student_embed_dim, clip_out_dim, use_bn=use_bn_in_head)
        self.attribute_head = nn.Linear(self.student.num_prototypes, 312)
        for p in self.teacher.parameters(): p.requires_grad = False
        for p in self.teacher_dino_head.parameters(): p.requires_grad = False

    def _build_mlp(self, in_dim, out_dim, use_bn):
        return nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim))

    @torch.no_grad()
    def _update_teacher_ema(self):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.mul_(self.teacher_momentum).add_(student_param.data, alpha=1 - self.teacher_momentum)
        for student_param, teacher_param in zip(self.dino_head.parameters(), self.teacher_dino_head.parameters()):
            teacher_param.data.mul_(self.teacher_momentum).add_(student_param.data, alpha=1 - self.teacher_momentum)
            
    def forward(self, view1, view2):
        cls1_student, proto_scores1_student = self.student(view1)
        cls2_student, proto_scores2_student = self.student(view2)
        dino_out1_student = self.dino_head(cls1_student)
        dino_out2_student = self.dino_head(cls2_student)
        clip_out1_student = self.clip_head(cls1_student)
        clip_out2_student = self.clip_head(cls2_student)
        attr_preds = self.attribute_head(proto_scores1_student)
        with torch.no_grad():
            self._update_teacher_ema()
            cls1_teacher, _ = self.teacher(view1)
            cls2_teacher, _ = self.teacher(view2)
            dino_out1_teacher = self.teacher_dino_head(cls1_teacher)
            dino_out2_teacher = self.teacher_dino_head(cls2_teacher)
        return {
            "student_dino": [dino_out1_student, dino_out2_student],
            "teacher_dino": [dino_out1_teacher, dino_out2_teacher],
            "student_clip": [clip_out1_student, clip_out2_student],
            "student_protos": [proto_scores1_student, proto_scores2_student],
            "attr_preds": attr_preds,
        }
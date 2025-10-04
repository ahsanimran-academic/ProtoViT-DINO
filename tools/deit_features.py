# tools/deit_features.py

import torch
import timm

def deit_tiny_patch_features(pretrained=True, **kwargs):
    """
    Loads the feature extractor of a pretrained DeiT-Tiny model.
    """
    model = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, **kwargs)
    model.eval()
    return model

def deit_small_patch_features(pretrained=True, **kwargs):
    """
    Loads the feature extractor of a pretrained DeiT-Small model.
    """
    model = timm.create_model('deit_small_patch16_224', pretrained=pretrained, **kwargs)
    model.eval()
    return model
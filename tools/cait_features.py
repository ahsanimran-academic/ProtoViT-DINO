# tools/cait_features.py

import torch
import timm

def cait_xxs24_224_features(pretrained=True, **kwargs):
    """
    Loads the feature extractor of a pretrained CaiT-XXS-24 model.
    """
    model = timm.create_model('cait_xxs24_224', pretrained=pretrained, **kwargs)
    model.eval()
    return model
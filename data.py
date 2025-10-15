# data.py
import torch
import json
import os
from torchvision import datasets, transforms
from PIL import Image


class SSLAugmentation:
    def __init__(self, img_size=224):
        # Using a slightly less aggressive crop scale, better for fine-grained datasets
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        # Return two different augmentations of the same image
        return self.transform(x), self.transform(x)

class SSLAttributeDataset(datasets.ImageFolder):
    """
    Dataset wrapper for SSL that returns two augmented views of an image,
    plus its corresponding attribute vector.
    """
    def __init__(self, root, attributes_json_path, transform=None):
        super().__init__(root, transform=None) # We handle transforms internally
        
        self.ssl_transform = transform if transform is not None else SSLAugmentation()
        
        print(f"Loading attributes from {attributes_json_path}")
        with open(attributes_json_path, 'r') as f:
            self.attributes_data = json.load(f)
            
    def __getitem__(self, index):
        path, _ = self.samples[index]  # Ignore original labels for SSL
        filename = os.path.basename(path)
        sample = self.loader(path)
        
        # Get augmented views
        view1, view2 = self.ssl_transform(sample)
        
        # Get attribute vector
        # If an image from your training folder is not in the json, return a zero vector
        attribute_vector = self.attributes_data.get(filename, [0] * 312)
        attributes = torch.tensor(attribute_vector, dtype=torch.float32)
        
        return view1, view2, attributes, index
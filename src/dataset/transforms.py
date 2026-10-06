import torch
import numpy as np
import cv2

class BasicTransform:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, image):
        # Resize using cv2
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        
        # Add channel dimension if necessary (H, W) -> (C, H, W)
        if len(image.shape) == 2:
            # ResNet18 expects 3 channels. Let's repeat the grayscale image 3 times.
            image = np.stack((image,)*3, axis=0)
        elif len(image.shape) == 3 and image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
            
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        
        # standard normalization for ResNet pretrained
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
            
        return image_tensor

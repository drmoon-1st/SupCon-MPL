# code from https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection.git
# code from https://github.com/kekmodel/MPL-pytorch.git

import cv2
import numpy as np
from PIL import Image, ImageFile
import random

import torch
from torch.utils.data import Dataset 
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# pytorch dataset class
class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, image_size, mode = 'train', contrast = False):
        self.x = images
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = images.shape[0]
        self.contrast = contrast
    
    def create_train_transforms(self, size):
        return transforms.Compose([
        transforms.RandomResizedCrop(size=self.image_size, scale=(0.5, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Resize((self.image_size, self.image_size)),
        transforms.ToTensor(),
        # normalize,
        ])
        
    def create_val_transform(self, size):
        return transforms.Compose([
        transforms.Resize((self.image_size, self.image_size)),
        transforms.ToTensor(),
        # normalize,
        ])
    
    def resize_origin_img(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        original_image = Image.open(self.x[index])

        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)
        
        if self.contrast:
            return transform(original_image), transform(original_image), self.y[index]
        else:
            augmented_image = transform(original_image)
            original_image = self.resize_origin_img()(original_image)
            return original_image, augmented_image, self.y[index]
        
    def __len__(self):
        return self.n_samples
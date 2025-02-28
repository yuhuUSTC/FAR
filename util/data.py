import numpy as np
import os
from pathlib import Path
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch
import io
import fsspec
from util.crop import center_crop_arr
import torchvision.transforms as transforms

            
###Custom dataset class for JournalDB
class JournalDBDataset(Dataset):
    def __init__(self, json_file, img_size=256):
        self.transform = self.get_transforms_image(name="center", image_size=(img_size, img_size))

        # Load the data from the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)  # Assumed to be a list of entries
            print(f"dateset length: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def get_transforms_image(self, name="center", image_size=(256, 256)):
        if name is None:
            return None
        elif name == "center":
            assert image_size[0] == image_size[1], "Image size must be square for center crop"
            transform = transforms.Compose([
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])
        elif name == "resize_crop":
            transform = transforms.Compose([
                    transforms.Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])
        else:
            raise NotImplementedError(f"Transform {name} not implemented")
        return transform

    def __getitem__(self, index):
        # Get the image path and description
        img_path, description, _, _ = self.data[index]
        #print(img_path)
        if img_path.endswith(('.png', '.jpeg', '.bmp', '.gif')):  # 如果有其他后缀，可以继续添加
            img_path = img_path.rsplit('.', 1)[0] + '.jpg'
        #img_path = os.path.join("/mnt/JournalDB/image_datasets/MidjourneyDB/train", img_path)
        img_path = os.path.join("/mnt/workspace/workgroup/yuhu/code/FAR_T2I/JournalDB/image_datasets/MidjourneyDB/train", img_path)

        try:
            img = Image.open(img_path).convert('RGB')  # Read and convert image
            if self.transform:
                img = self.transform(img)
            return img, description  # Return the image and its description
        except:
            img_path, description, _, _ = self.data[1]
            if img_path.endswith(('.png', '.jpeg', '.bmp', '.gif')):  # 如果有其他后缀，可以继续添加
                img_path = img_path.rsplit('.', 1)[0] + '.jpg'
            #img_path = os.path.join("/mnt/JournalDB/image_datasets/MidjourneyDB/train", img_path)
            img_path = os.path.join("/mnt/workspace/workgroup/yuhu/code/FAR_T2I/JournalDB/image_datasets/MidjourneyDB/train", img_path)
            img = Image.open(img_path).convert('RGB')  # Read and convert image
            if self.transform:
                img = self.transform(img)
            return img, description  # Return the image and its description

   

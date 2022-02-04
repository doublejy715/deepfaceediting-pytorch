from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import random

import numpy as np
# step 1
class Sketch_Encoder_Dataset(Dataset):
    def __init__(self, data_path):
        self.dataset = sorted(glob.glob(data_path+'geo/*.*'))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        Xs = Image.open(image_path).convert("L")
        return self.transforms(Xs)

    def __len__(self):
        return len(self.dataset)
# step 2
class Img_Encoder_Dataset(Dataset):
    def __init__(self,data_path):
        self.img_dataset = sorted(glob.glob(data_path+'/image/*.*'))
        self.geo_dataset = sorted(glob.glob(data_path+'/geo/*.*'))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        img = Image.open(self.img_dataset[idx])
        geo = Image.open(self.geo_dataset[idx]).convert("L")
        return self.transforms(img), self.transforms(geo)

    def __len__(self):
        return len(self.img_dataset)

# step 3,4
class Dataset(Dataset):
    def __init__(self,data_path):
        self.img_dataset = glob.glob(data_path+'/image/*.*')
        self.geo_dataset = glob.glob(data_path+'/image/*.*')
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
    # 랜덤
    def __getitem__(self, idx):
        img = Image.open(self.img_dataset[idx])
        geo = Image.open(random.choice(self.geo_dataset))
        return self.transforms(img), self.transforms(geo)

    def __len__(self):
        return len(self.img_dataset)

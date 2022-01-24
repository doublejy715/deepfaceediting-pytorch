from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob

import numpy as np

class Sketch_Encoder_Dataset(Dataset):
    def __init__(self, data_path):
        self.dataset = sorted(glob.glob(data_path+'sketch/*.*'))
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

class Img_Encoder_Dataset(Dataset):
    def __init__(self,data_path):
        self.img_dataset = sorted(glob.glob(data_path+'/image/*.*'))
        self.sketch_dataset = sorted(glob.glob(data_path+'/sketch/*.*'))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        img = Image.open(self.img_dataset[idx])
        sketch = Image.open(self.sketch_dataset[idx]).convert("L")
        return self.transforms(img), self.transforms(sketch)

    def __len__(self):
        return len(self.img_dataset)

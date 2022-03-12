import os
import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Dataset(Dataset):
    def __init__(self, img_path, isMaster):
        self.datasets = glob.glob(img_path+'/image/*.*')
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        if isMaster:
            print(f"Dataset of {str(self.__len__())} images constructed.")

    def __getitem__(self, idx):
        app = Image.open(self.datasets[idx])
        geo = Image.open(random.choice(self.datasets))
        return self.transforms(app), self.transforms(geo)

    def __len__(self):
        return len(self.datasets)


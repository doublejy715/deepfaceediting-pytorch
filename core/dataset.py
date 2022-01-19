from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob

class SketchDataset(Dataset):
    def __init__(self, data_path_list):
        datasets = []
        for data_path in glob.glob(f'{data_path_list}/*.*g'):
            datasets.append(data_path)
        self.datasets = datasets
        self.transforms = transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        image_path = self.datasets[idx]
        Xs = Image.open(image_path).convert("RGB").resize((256,256))
        return self.transforms(Xs)

    def __len__(self):
        return len(self.datasets)


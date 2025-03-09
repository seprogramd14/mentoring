import os
import pandas as pd
from PIL import Image
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, label_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.label_file = pd.read_csv(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.label_file.iloc[index, 0])
        img = Image.open(img_path).convert("RGB")
        label = self.label_file.iloc[index, 1]

        if self.transform:
            img = self.transform(img)

        return img, label

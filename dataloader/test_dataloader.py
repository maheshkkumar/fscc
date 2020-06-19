import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_data(img_path):
    gt_path = img_path.replace('.jpg', '.csv').replace('frames', 'csvs')
    img = Image.open(img_path).convert('RGB')
    target = np.loadtxt(gt_path, delimiter=',')
    return img, target


class TestDataset(Dataset):
    def __init__(self, dataset):

        self.root = 'data_set_path'
        self.images = self.load_images()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])
        self.dataset = dataset

    def load_images(self):
        folders = [os.path.join(self.root, img) for img in os.listdir(self.root)]
        images = []
        for folder in folders:
            imgs = [os.path.join(folder, img) for img in os.listdir(folder)]
            images += imgs
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]

        img, target = load_data(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

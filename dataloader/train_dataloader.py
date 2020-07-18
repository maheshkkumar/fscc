import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GetDataLoader(object):
    def __init__(self):
        pass

    def get_data(self, task, batch_size=1, mode='train'):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        d_task = TrainDataset(task, dataset=task.dataset, transform=transform, mode=mode)
        dataloader = DataLoader(d_task, batch_size=batch_size, num_workers=10, pin_memory=True)

        return dataloader


class TrainDataset(Dataset):
    def __init__(self, task, dataset, mode, transform=None):

        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.task = task
        self.data_path = self.task.data_path

        if self.mode == 'train':
            self.imgs = self.task.train_images
        else:
            self.imgs = self.task.validation_images

    def __len__(self):
        return len(self.imgs)

    def load_data(self, img_path):
        gt_path = img_path.replace('.jpg', '.csv').replace('frames', 'csvs')
        img = Image.open(img_path).convert('RGB')
        target = np.loadtxt(gt_path, delimiter=',')

        if self.mode == 'train':
            if random.random() > 0.8:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # resizing target based on the dataset
            target = cv2.resize(target, (target.shape[1] / 8, target.shape[0] / 8), interpolation=cv2.INTER_CUBIC) * 64

        return img, target

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img, target = self.load_data(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

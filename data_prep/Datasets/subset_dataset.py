import torch
import torch.utils.data
import torchvision
from torchvision.datasets.folder import default_loader
import numpy as np
import glob
import os


def list_dir(dir: str):
    return glob.glob(os.path.join(dir, '*'))


def choose_pics(id: str, num_pics: int = 0):
    id_pics = list_dir(id)
    if num_pics == 0:
        num_pics = len(id_pics)
    return np.random.choice(id_pics, size=num_pics, replace=False)


class RandomSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str,
                 ids_list: list,
                 source_transform=None,
                 target_transform=None,
                 num_pics_per_id: int = 0,
                 loader=default_loader):
        self.dir = dir
        self.loader = loader
        self.transform = source_transform
        self.target_transform = target_transform
        self.ids = ids_list
        self.samples = []
        for i in range(len(self.ids)):
            pics = choose_pics(os.path.join(self.dir, self.ids[i]), num_pics_per_id)
            for pic in pics:
                self.samples.append((pic, i))


    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
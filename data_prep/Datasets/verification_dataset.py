import os
import torch
from torch.utils.data import Dataset
from torchvision import io
from typing import Optional


class VerificationDataset(Dataset):
    """
    A dataset retrieving pairs of images, along with their 'keys' and verification label if exists (same=1, diff=0)
    """
    def __init__(self, dir_path: str, pairs_list: list, labels: Optional[list] = None, transforms: Optional[object] = None):
        self.__dir_path = dir_path
        self.__pairs_list = pairs_list
        self.__labels = labels
        self.__transforms = transforms

    def __load_image(self, idx, pair_idx):
        im_path = os.path.join(self.__dir_path,
                                self.__pairs_list[idx][pair_idx])
        image = io.read_image(im_path)

        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        """
        return t1, t2, key1, key2, same_label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.__labels:
            label = self.__labels[idx]

        return (self.__load_image(idx, 0),
                self.__load_image(idx, 1),
                self.__pairs_list[idx][0],
                self.__pairs_list[idx][1],
                label)

    def __len__(self):
        return len(self.__pairs_list)

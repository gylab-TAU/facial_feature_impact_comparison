from typing import Tuple, Any

import torch
import torchvision


class ActivationsDatasets(torchvision.datasets.ImageFolder):
    def __init__(self, path, transforms, target_transforms):
        super(ActivationsDatasets, self).__init__(path, transforms, target_transform=target_transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, file_path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target + 8749, path

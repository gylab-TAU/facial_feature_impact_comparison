from data_prep.Datasets.subset_dataset import RandomSubsetDataset
from data_prep.image_loader import ImageLoader
import torchvision.datasets as datasets
import glob
import os
import numpy as np


def list_dir(dir: str):
    return glob.glob(os.path.join(dir, '*'))


def filter_ids(ids:list, num_pics: int):
    filtered_ids = []
    for id in ids:
        id_pics = list_dir(id)
        if len(id_pics) >= num_pics:
            filtered_ids.append(id)
    return filtered_ids


def choose_ids(filtered_ids: list, num_ids: int):
    return np.random.choice(filtered_ids, size=num_ids, replace=False)


class RandomExperimentImageLoader(ImageLoader):
    def __init__(self, im_size, post_crop_size, dataset_mean, dataset_std, target_transform=None):
        super(RandomExperimentImageLoader, self).__init__(im_size, post_crop_size, dataset_mean, dataset_std, target_transform)

    def load_ids_dataset(self, dir_path: str, ids:list, center_crop=True):
        if center_crop:
            return RandomSubsetDataset(dir_path, ids,  self.center_crop_tt, target_transform=self.target_transform)
        return RandomSubsetDataset(dir_path, ids, self.random_crop_tt, target_transform=self.target_transform)

    def load_random_dataset(self, dir_path, num_ids: int = 0, num_pics=0, center_crop=True):
        all_ids = list_dir(dir_path)
        filtered_ids = filter_ids(all_ids, num_pics)
        ids = [os.path.relpath(id, dir_path) for id in choose_ids(filtered_ids, num_ids)]
        if center_crop:
            return RandomSubsetDataset(dir_path, ids, self.center_crop_tt, target_transform=self.target_transform, num_pics_per_id=num_pics)
        return RandomSubsetDataset(dir_path, ids, self.random_crop_tt, target_transform=self.target_transform, num_pics_per_id=num_pics)
from os import path
from glob import glob
from typing import List, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd

import argparse
import os
import shutil


def rename_path(pth, mapping_path: str = '/home/administrator/Documents/imagenet_classes.csv'):
    mapping = pd.read_csv(mapping_path, index_col='cls')
    cls = os.path.basename(os.path.dirname(pth))
    content = mapping.loc[cls, 'content']
    return pth.replace(cls, content)


def transfer_datapoints(dest_dataset_loc, source_path, data_points, renaming=False):
    pbar = tqdm(data_points)
    for point in pbar:
        dest_point = os.path.join(dest_dataset_loc, os.path.relpath(point, source_path))
        if renaming:
            dest_point = rename_path(dest_point)
        os.makedirs(os.path.dirname(dest_point), exist_ok=True)
        try:
            os.symlink(os.path.abspath(point), dest_point)
        except OSError:
            pbar.set_description("Encountered error on point. Copying...")
            shutil.copyfile(point, dest_point)


def sample_dataset(num_cls: int, num_img_per_cls: int, dataset_path: str) -> Dict[str, List[str]]:
    cls = glob(path.join(dataset_path, '*'))
    chosen_cls = np.random.choice(
        cls,
        size=num_cls,
        replace=False)

    chosen_imgs = []
    for cls in chosen_cls:
        chosen_imgs = chosen_imgs + [img for img in np.random.choice(
            glob(path.join(cls, '*')),
            size=num_img_per_cls,
            replace=False)]

    return chosen_imgs


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="/home/ssd_storage/datasets/celebA_crops")
    parser.add_argument("--samples_dest", type=str, default='/home/ssd_storage/datasets/objects_faces_rdm')
    parser.add_argument("--num_cls", type=int, default=50)
    parser.add_argument("--num_imgs_per_cls", type=int, default=10)
    parser.add_argument("--should_imagenet_rename", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    samples = sample_dataset(args.num_cls, args.num_imgs_per_cls, args.dataset_path)
    print(samples)
    transfer_datapoints(args.samples_dest, args.dataset_path, samples, renaming=args.should_imagenet_rename)

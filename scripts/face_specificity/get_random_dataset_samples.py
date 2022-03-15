from argparse import ArgumentParser, Namespace
from glob import glob
from numpy import random
from os import path, makedirs
from typing import List

import shutil


def get_dir_content(dir: str) -> List[str]:
    return glob(path.join(dir, '*'))


def sample_dir_content(dir: str, num_samples: int) -> List[str]:
    all_content = get_dir_content(dir)
    print(all_content)
    chosen_samples = random.choice(all_content, num_samples, replace=False)
    return chosen_samples


def copy_data(src_dir: str, dest_dir: str, chosen_samples: List[str]) -> None:
    for sample in chosen_samples:
        dest_point = path.join(dest_dir, path.relpath(sample, src_dir))
        print(dest_point)
        makedirs(path.dirname(dest_point), exist_ok=True)
        shutil.copyfile(sample, dest_point)


def sample_dataset(src_dir: str, dest_dir: str, num_cls: int, num_imgs: int, chosen_cls: List[str] = None) -> None:
    all_ds = get_dir_content(src_dir)
    for ds in all_ds:
        if chosen_cls is None:
            chosen_cls = sample_dir_content(path.join(ds, 'images'), num_cls)
        for cls in chosen_cls:
            chosen_imgs = sample_dir_content(cls, num_imgs)
            copy_data(src_dir, dest_dir, chosen_imgs)


def sample_dataset(ds_dir: str, dest_dir: str, num_cls: int, num_imgs: int, chosen_cls: List[str] = None) -> None:
    if chosen_cls is None:
        chosen_cls = sample_dir_content(ds_dir, num_cls)
    for cls in chosen_cls:
        chosen_imgs = sample_dir_content(cls, num_imgs)
        copy_data(ds_dir, dest_dir, chosen_imgs)


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--all_datasets_dir', type=str,
                        default="/home/ssd_storage/datasets/processed/num_classes/260_inanimate_imagenet_num-classes_260/train")
    parser.add_argument('--dest_dir', type=str,
                        default="/home/ssd_storage/experiments/birds/datasets_samples/imagenet_inanimates")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--chosen_cls_delimited', type=str, default='n04507155,n02951358,n03792782,n03100240,n04536866')
    parser.add_argument('--num_imgs_per_cls', type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    chosen_cls = [path.join(args.all_datasets_dir, item) for item in args.chosen_cls_delimited.split(',')]
    print(chosen_cls)
    sample_dataset(args.all_datasets_dir, args.dest_dir, args.num_classes, args.num_imgs_per_cls, chosen_cls=chosen_cls)
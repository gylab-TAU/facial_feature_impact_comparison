from glob import glob
from typing import List, Dict
from argparse import ArgumentParser
import numpy as np
from os import path
import os
import shutil


def transfer_datapoints(dest_dataset_loc: str, source_path: str, data_points: List[str]) -> None:
    """
    Create a symlink (or copy, if symlink cannot be made) for all data_points in source_path to the new dest_dataset_loc

    :dest_dataset_loc: The destination directory, (where to move the files to)
    :source_path: The source directory containing the files
    :data_points: The absolute path of all files to copy
    """
    # pbar = tqdm(data_points)
    for point in data_points:
        dest_point = os.path.join(dest_dataset_loc, os.path.relpath(point, source_path))
        os.makedirs(os.path.dirname(dest_point), exist_ok=True)
        try:
            os.symlink(os.path.abspath(point), dest_point)
        except OSError:
            # pbar.set_description("Encountered error on point. Copying...")
            shutil.copyfile(point, dest_point)


def globall(data_dir: str) -> List[str]:
    """
    get all content from a directory
    """
    return glob(path.join(data_dir, '*'))


def get_cls_names(ds_dir: str) -> List[str]:
    """
    Get all classes names from ds_dir (taken from the 'train' dir, should be the same classes as in 'val')
    """
    train_dir = path.join(ds_dir, 'train')
    cls = globall(train_dir)
    cls = [path.relpath(cl, train_dir) for cl in cls]

    return cls


def choose_random_cls(ds_dir: str, num_cls: int) -> List[str]:
    """
    Choose random n :num_cls: from the :ds_dir:/train directory
    """
    cls = get_cls_names(ds_dir)
    return np.random.choice(cls, num_cls, replace=False)


def copy_phase_cls(src_ds: str, dest_ds: str, phase: str, cls: List[str]) -> None:
    """
    Copy all content from src_ds/phase/cl to dest_ds/phase/cl for cl in cls
    """
    for cl in cls:
        src_phase = path.join(src_ds, phase, cl)
        data = globall(src_phase)
        dest_phase = path.join(dest_ds, phase, cl)

        transfer_datapoints(dest_dataset_loc=dest_phase, source_path=src_phase, data_points=data)


def copy_dataset(src_ds: str, dest_ds: str, cls: List[str]) -> None:
    """
    Copy all content from all classes in cls, from the train/val directories in src_ds to the same directories in phase
    """
    copy_phase_cls(src_ds, dest_ds, 'train', cls)
    copy_phase_cls(src_ds, dest_ds, 'val', cls)


def get_cl_phase_imgs(cl_phase_dir: str) -> List[str]:
    """
    Get the names of images in a cl dir
    """
    imgs = globall(cl_phase_dir)
    imgs = [path.relpath(img, cl_phase_dir) for img in imgs]
    return imgs


def gather_all_img_names(src_ds: str) -> Dict[str, List[str]]:
    """
    For every class in src_ds, get all images in both train and val dirs.
    """
    cls_names = get_cls_names(src_ds)
    cls_imgs = {}
    for cl in cls_names:

        train_imgs = get_cl_phase_imgs(path.join(src_ds, 'train', cl))
        val_imgs = get_cl_phase_imgs(path.join(src_ds, 'val', cl))

        cls_imgs[cl] = train_imgs + val_imgs

    return cls_imgs


def get_untrained_imgs(untrained_dir: List[str], trained_dir: List[str], num_imgs: int) -> List[str]:
    """
    Given a dir with untrained data, and a dir with trained data, get a list of untrained imgs of size num_imgs
    """
    untrained_imgs = list(set(untrained_dir) - set(trained_dir))
    return np.random.choice(untrained_imgs, num_imgs, replace=False)


def get_classes_untrained(cls2imgs: Dict[str, List[str]], untrained_dir: str, num_imgs: int) -> Dict[str, List[str]]:
    """
    Get num_imgs images from the untrained dir belonging to every class in the mapping
    :cls2imgs: mapping the cls name to the imgs used for training
    :untrained_dir: directory containing the fill cls dirs
    :num_imgs: the number of imgs to get for every class
    :returns: A mapping between the cls name to the untrained imgs
    """
    untrained_cls2imgs = {}

    for cl in cls2imgs:
        full_cls = get_cl_phase_imgs(path.join(untrained_dir, cl))
        untrained_cls2imgs[cl] = get_untrained_imgs(full_cls, cls2imgs[cl], num_imgs)

    return untrained_cls2imgs


def enrich_dataset(src_ds: str, target_ds: str, enrichment_cls2imgs: Dict[str, List[str]]) -> None:
    for cl in enrichment_cls2imgs:
        src_phase = src_ds # path.join(src_ds, phase)
        dest_phase = path.join(target_ds, 'val')
        abs_path_images = [path.join(src_phase, cl, img) for img in enrichment_cls2imgs[cl]]
        transfer_datapoints(dest_dataset_loc=dest_phase, source_path=src_phase, data_points=abs_path_images)


def create_enriched_dataset(trained_ds: str, untrained_ds: str, target_ds: str, num_cls: int, num_enriched: int) -> None:

    # Create a copy of num_cls classes from the trained dataset
    chosen_cls = choose_random_cls(ds_dir=trained_ds, num_cls=num_cls)
    copy_dataset(src_ds=trained_ds, dest_ds=target_ds, cls=chosen_cls)
    if (num_enriched > 0) and (untrained_ds is not None):
        # add the enrichment images
        cls2imgs = gather_all_img_names(src_ds=target_ds)
        untrained_cls2imgs = get_classes_untrained(cls2imgs=cls2imgs, untrained_dir=untrained_ds, num_imgs=num_enriched)

        enrich_dataset(src_ds=untrained_ds, target_ds=target_ds, enrichment_cls2imgs=untrained_cls2imgs)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('trained_ds', type=str)
    parser.add_argument('--untrained_ds', type=str, default=None)
    parser.add_argument('target_ds', type=str)
    parser.add_argument('num_cls', type=int)
    parser.add_argument('--num_enriched', type=int, default=0)
    args = parser.parse_args()
    assert (args.num_cls > 0)
    return args


if __name__ == '__main__':
    args = get_args()
    create_enriched_dataset(
        trained_ds=args.trained_ds,
        untrained_ds=args.untrained_ds,
        target_ds=args.target_ds,
        num_cls=args.num_cls,
        num_enriched=args.num_enriched)

from argparse import ArgumentParser
from glob import glob
from os import path
from typing import List

import csv
import pandas as pd


def get_dir_content(dir: str) -> List[str]:
    return glob(path.join(path.join(dir, '*')))


def get_all_verification_pairs(dataset_path: str) -> pd.DataFrame:
    df = {'im1': [], 'im2': [], 'same': []}

    classes = get_dir_content(dataset_path)

    # for each class in the dataset:
    for k, cls1 in enumerate(classes):
        cls_imgs = get_dir_content(cls1)
        num_cls_imgs = len(cls_imgs)

        # for each distinct pair from a class add the pairs and mark them as same:
        for i in range(num_cls_imgs):
            for j in range(i+1, num_cls_imgs):
                df['im1'].append(path.relpath(cls_imgs[i], dataset_path))
                df['im2'].append(path.relpath(cls_imgs[j], dataset_path))
                df['same'].append(1)

        # for each distinct pair of classes
        for i in range(k+1, len(classes)):
            cls2 = classes[i]
            cls2_imgs = get_dir_content(cls2)
            # for each pair of images belonging to different classes, add the pair and mark them as diff:
            for im1 in cls_imgs:
                for im2 in cls2_imgs:
                    df['im1'].append(path.relpath(im1, dataset_path))
                    df['im2'].append(path.relpath(im2, dataset_path))
                    df['same'].append(0)

    return pd.DataFrame(df)

def get_args():
    parser = ArgumentParser()
    # parser.add_argument('--ds_dir', type=str, default="/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_bird_species_rdm_30_10_num-classes_30_{'images': 10}/images")
    # parser.add_argument('--output_path', type=str, default="/home/ssd_storage/experiments/birds/verification/species_verification_pairs.txt")
    parser.add_argument('--ds_dir', type=str, default="/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_individual_birds_rdm_30_10_num-classes_30_{'images': 10}/images")
    parser.add_argument('--output_path', type=str, default="/home/ssd_storage/experiments/birds/verification/sociable_weavers_verification_pairs.txt")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    verification_pairs = get_all_verification_pairs(args.ds_dir)
    verification_pairs.to_csv(args.output_path, sep=' ', header=False, index=False)
    print(args.output_path)
    print(args.ds_dir)
    # df = pd.read_csv(args.output_path, sep=' ', index_col=False, names=['im1', 'im2', 'same'])
    # print(df['same'].to_list())



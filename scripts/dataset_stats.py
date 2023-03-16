from typing import Tuple
import numpy as np
from glob import glob
from argparse import ArgumentParser
from os import path


def load_ds_stats(ds_path: str) -> Tuple[float, float, float]:
    cls = glob(path.join(ds_path, '*'))
    img_count = [len(glob(path.join(cl, '*'))) for cl in cls]
    return float(np.min(img_count)), float(np.max(img_count)), float(np.mean(img_count))


def get_args():
    parser = ArgumentParser()
    parser.add_argument('ds_path', type=str,
                        help='location to the dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    n_min, n_max, n_mean = load_ds_stats(args.ds_path)
    print(f'min: {n_min}, max: {n_max}, mean: {n_mean}')

import glob
import argparse
import os
import string

import numpy as np
import numpy.random as npr
from data_prep.util import transfer_datapoints


def divide_ds(classes, num_divs: int):
    divisions = []
    div_size = len(classes) // num_divs
    for i in range(num_divs-1):
        divisions.append(npr.choice(classes, div_size, replace=False))
        classes = np.setdiff1d(classes, divisions[i])
    divisions.append(classes)
    return divisions


def copy_divisions(divisions, src_dir: str, dest_dir: str):
    for div, div_name in zip(divisions, list(string.ascii_uppercase)):
        for cl in div:
            cl_src = os.path.join(src_dir, cl)
            cl_dest = os.path.join(dest_dir, div_name, cl)
            cl_data = glob.glob(os.path.join(cl_src, '*'))
            transfer_datapoints(cl_dest, cl_src, cl_data)


def get_classes(src_dir):
    cls = glob.glob(os.path.join(src_dir, '*'))
    return [os.path.relpath(cl, src_dir) for cl in cls]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", type=str, default="/home/administrator/experiments/familiarity/dataset/finetuning_dataset_fixed")
    parser.add_argument("--dest_dir", type=str, default="/home/administrator/experiments/familiarity/dataset/divided_finetuning_fixed")
    parser.add_argument("--num_divs", type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    cls = get_classes(args.src_dir)
    divs = divide_ds(cls, args.num_divs)
    copy_divisions(divs, args.src_dir, args.dest_dir)

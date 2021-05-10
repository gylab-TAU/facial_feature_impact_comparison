import os
from pathlib import Path
import argparse
import glob
import tqdm
import pandas as pd


def get_pairs_lists(root, depth):
    root_glob = root
    for i in range(depth):
        root_glob = os.path.join(root_glob, '*')
    print(f"reading pairs from {root_glob}")
    return glob.glob(root_glob)



def verify_img(img, imgs_set):
    if img in imgs_set:
        print(f"multiple records of {img} in list")
    else:
        imgs_set.add(img)
    return imgs_set


def verify_unique_files(list_df, label):
    imgs_set = set()
    for i in tqdm.tqdm(range(len(list_df)), desc=label):
        img1 = list_df.iloc[i][0]
        img2 = list_df.iloc[i][1]
        imgs_set = verify_img(img1, imgs_set)
        imgs_set = verify_img(img2, imgs_set)
        if 'same' in label:
            if img1.split('/')[0] != img2.split('/')[0]:
                print(f"Diff where there should be same! {img1}, {img2}")
        if 'diff' in label:
            if img1.split('/')[0] == img2.split('/')[0]:
                print(f"Same  where there should be Diff! {img1}, {img2}")


def verify_multiple_pairs_lists(pairs_lists):
    for pairs in pairs_lists:
        cols = ['img1','img2']
        print(f"Checking {pairs}...")
        same = pairs + "_same.txt"
        diff = pairs + "_diff.txt"

        same_df = pd.read_csv(same, sep=' ', header=None, index_col=False)
        verify_unique_files(same_df, same)

        diff_df = pd.read_csv(diff, sep=' ', header=None, index_col=False)
        verify_unique_files(diff_df, diff)

        list_df = same_df.append(diff_df)
        verify_unique_files(list_df, pairs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs_dir", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/image_pairs_lists/mutualy_exclusive")
    parser.add_argument('--depth', type=int,
                        default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # files = get_pairs_lists(args.pairs_dir, args.depth)
    files = [os.path.join(args.pairs_dir, div) for div in ['A', 'B', 'C']]
    verify_multiple_pairs_lists(files)
    print("Done")
import os
from pathlib import Path
import argparse
import glob
import tqdm


def get_imgs_links(root, depth):
    root_glob = root
    for i in range(depth):
        root_glob = os.path.join(root_glob, '*')
    print(f"reading imgs from {root_glob}")
    return glob.glob(root_glob)


def iterate_divided_dataset(links):
    imgs_map = {}

    for link in tqdm.tqdm(links, desc="Imgs"):
        abs_img_path = Path(link).resolve()
        print('abs_img_path: ', abs_img_path)
        print('imgs_map: ', imgs_map)
        if abs_img_path in imgs_map:
            print(f"Two links to the same img: {link} and {imgs_map[abs_img_path]} -> {abs_img_path}")
        else:
            imgs_map[abs_img_path] = link


def compare_dsets(datasets_roots, depths):
    links = []
    for root, depth in zip(datasets_roots, depths):
        links = links + get_imgs_links(root, depth)
    iterate_divided_dataset(links)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir1", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/processed_finetuning_dataset/phase_perc_size/")
    parser.add_argument("--dir2", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/processed_pretraining_dataset/phase_perc_size/pretraining_fixed_{'train': 0.7, 'val': 0.2, 'test': 0.1}/")
    parser.add_argument('--depth1', type=int,
                        default=4)
    parser.add_argument('--depth2', type=int,
                        default=3)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    compare_dsets([args.dir1, args.dir2], [args.depth1, args.depth2])
    print("Done")

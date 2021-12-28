from PIL import Image
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

import os


def save_inverted(im_path: str, src_dir: str, output_dir: str):
    im_cls_title = os.path.relpath(im_path, src_dir)
    output_path = os.path.join(output_dir, im_cls_title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    orig = Image.open(im_path)
    inverted = orig.transpose(Image.FLIP_TOP_BOTTOM)

    inverted.save(output_path)

    orig.close()
    inverted.close()


def invert_dataset(src_dir: str, output_dir: str):
    for img in tqdm(glob(os.path.join(src_dir, '*', '*'))):
        save_inverted(img, src_dir, output_dir)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--src_dir', type=str, default="/home/ssd_storage/datasets/processed/random_2_imgs_from_80_cls/familiar_birds_rdm_num-classes_80_{'images': 2}/images")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dirs = [
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_bird_species_rdm_30_10_num-classes_30_{'images': 10}/images",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_individual_birds_rdm_30_10_num-classes_30_{'images': 10}/images",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_faces_rdm_30_10_num-classes_30_{'images': 10}/images",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_inanimate_rdm_30_10_num-classes_30_{'images': 10}/images"
    ]
    for src_dir in dirs:
        inverted_output_dir = f'{src_dir}_inverted'
        os.makedirs(inverted_output_dir, exist_ok=True)
        invert_dataset(src_dir, inverted_output_dir)
    # inverted_output_dir = f'{args.src_dir}_inverted'
    # os.makedirs(inverted_output_dir, exist_ok=True)
    # invert_dataset(args.src_dir, inverted_output_dir)

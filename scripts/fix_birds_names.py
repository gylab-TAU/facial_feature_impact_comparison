from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm
import os


def rename(path):
    dir = os.path.dirname(path)
    cls = os.path.basename(dir)
    im_name = os.path.basename(path)
    new_fn = os.path.join(dir, f'{cls}_{im_name}')
    os.rename(path, new_fn)


def rename_dir(dir):
    imgs = glob(os.path.join(dir, '*', '*'))
    print(len(imgs))
    for img in tqdm(imgs):
        rename(img)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default="/home/ssd_storage/datasets/processed/random_2_imgs_from_80_cls/familiar_birds_rdm_num-classes_80_{'images': 2}/images_inverted")

    return parser.parse_args()


if __name__ == '__main__':
    # args = get_args()
    # rename_dir(args.dir)
    dirs = [
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_bird_species_rdm_30_10_num-classes_30_{'images': 10}/images",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_individual_birds_rdm_30_10_num-classes_30_{'images': 10}/images",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_faces_rdm_30_10_num-classes_30_{'images': 10}/images",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_inanimate_rdm_30_10_num-classes_30_{'images': 10}/images",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_bird_species_rdm_30_10_num-classes_30_{'images': 10}/images_inverted",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_individual_birds_rdm_30_10_num-classes_30_{'images': 10}/images_inverted",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_faces_rdm_30_10_num-classes_30_{'images': 10}/images_inverted",
        "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_inanimate_rdm_30_10_num-classes_30_{'images': 10}/images_inverted"
    ]
    for dir in dirs:
        rename_dir(dir)

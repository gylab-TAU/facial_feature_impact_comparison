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
    # args = get_args()
    # dirs = [
    #     # "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_bird_species_rdm_30_10_num-classes_30_{'images': 10}/images",
    #     # "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_individual_birds_rdm_30_10_num-classes_30_{'images': 10}/images",
    #     # "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_faces_rdm_30_10_num-classes_30_{'images': 10}/images",
    #     # "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_inanimate_rdm_30_10_num-classes_30_{'images': 10}/images"
    #         "/home/ssd_storage/datasets/processed/random_10_imgs_from_30_cls/familiar_max_img_faces_rdm_30_10_num-classes_30_{'images': 10}/images"
    # ]
    # dirs = {
    #     'species': '/home/ssd_storage/datasets/processed/verification_datasets/bird_species',
    #     'faces': "/home/ssd_storage/datasets/processed/30_max_imgs_vggface2_mtcnn white_list_{'train': 0.8, 'val': 0.2}/val",
    #     'sociable_weavers': "/home/ssd_storage/datasets/processed/phase_perc_size/individual_birds_single_species_{'train': 0.8, 'val': 0.2}/val",
    #     'inanimate_objects': "/home/ssd_storage/datasets/processed/num_classes/30_cls_inanimate_imagenet/val"
    # }
    dirs = {
        'Upright': '/home/ssd_storage/datasets/Thatcher/Upright/',
    }
    for label in dirs:
        src_dir = dirs[label]
        print(src_dir)
        # inverted_output_dir = f'/home/ssd_storage/datasets/processed/verification_datasets/{label}_inverted'
        inverted_output_dir = f'/home/ssd_storage/datasets/Thatcher/Inverted/'
        os.makedirs(inverted_output_dir, exist_ok=True)
        invert_dataset(src_dir, inverted_output_dir)
    # inverted_output_dir = f'{args.src_dir}_inverted'
    # os.makedirs(inverted_output_dir, exist_ok=True)
    # invert_dataset(args.src_dir, inverted_output_dir)

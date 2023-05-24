from PIL import Image
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import fnmatch
import os


def save_inverted(im_path: str, src_dir: str, output_dir: str):
    im_cls_title = os.path.relpath(im_path, src_dir)
    output_path = os.path.join(output_dir, 'flipped_'+im_cls_title)
    print (im_cls_title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    orig = Image.open(im_path)
    inverted = orig.transpose(Image.FLIP_TOP_BOTTOM)

    inverted.save(output_path)

    orig.close()
    inverted.close()


def invert_dataset(src_dir: str, output_dir: str):

    for img in tqdm(glob(os.path.join(src_dir, *,'*.JPG'))):
        save_inverted(img, src_dir, output_dir)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--src_dir', type=str)

    return parser.parse_args()




if __name__ == '__main__':
    dirs = {'/home/ssd_storage/datasets/MR/Monkeys/raw'
    }
    for label in dirs:
        src_dir = label
        inverted_output_dir = f'/home/ssd_storage/datasets/MR/Monkeys/raw'
        os.makedirs(inverted_output_dir, exist_ok=True)
        invert_dataset(src_dir, inverted_output_dir)





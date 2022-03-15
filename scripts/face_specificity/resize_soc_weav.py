from glob import glob
from os import path, makedirs
from typing import List

from torchvision import transforms
from PIL import Image
from argparse import ArgumentParser


def resize_imgs(input_imgs: List[str], output_imgs: List[str]) -> None:
    resize = transforms.Resize([224, 224])
    for inp, out in zip(input_imgs, output_imgs):
        inp_im = Image.open(inp)
        out_im = resize(inp_im)
        makedirs(path.dirname(out) ,exist_ok=True)
        out_im.save(out)


def make_out_paths(input_imgs: List[str], src_dir: str, out_dir: str) -> List[str]:
    out_paths = []
    for inp in input_imgs:
        rel = path.relpath(inp, src_dir)
        out = path.join(out_dir, rel)
        out_paths.append(out)
    return out_paths


def copy_resize_dir(src_dir: str, out_dir: str) -> None:
    input_imgs = glob(path.join(src_dir, '*', '*'))
    out_imgs = make_out_paths(input_imgs, src_dir, out_dir)
    resize_imgs(input_imgs, out_imgs)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--src_dir',
                        default="/home/ssd_storage/experiments/birds/datasets_samples/imagenet_inanimates" ,
                        type=str)
    parser.add_argument('--out_dir',
                        default="/home/ssd_storage/experiments/birds/datasets_samples/resized_imagenet",
                        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    copy_resize_dir(args.src_dir, args.out_dir)
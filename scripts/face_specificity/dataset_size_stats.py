from glob import glob
from os import path
import numpy as np


def get_stats(dir: str) -> None:
    cls = glob(path.join(dir, '*'))
    imgs = [len(glob(path.join(cl, '*'))) for cl in cls]
    print(f'min={min(imgs)}, max={max(imgs)}, avg={np.mean(imgs)}')


if __name__ == '__main__':
    ds = [
        '/home/ssd_storage/datasets/processed/30_max_imgs_vggface2_mtcnn white_list',
        '/home/KAD/project/datasets/260_birds_consolidated',
        '/home/ssd_storage/datasets/individual_birds/Cropped_pictures/IndividuaID/sociable_weavers/Train',
    ]

    for d in ds:
        get_stats(d)

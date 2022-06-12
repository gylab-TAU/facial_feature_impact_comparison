import os
import glob
import argparse
import numpy.random as npr
import pandas as pd
import tqdm


def globall(path):
    return glob.glob(os.path.join(path, '*'))


def relall(path_lst, rel):
    return [os.path.relpath(p, rel) for p in path_lst]


def get_cl_imgs(cl, root):
    return relall(globall(cl), root)


def generate_division_pairs(classes_dir: str, num_pairs: int):
    cls = globall(classes_dir)
    cols = ['img1', 'img2']
    same_pairs = {'img1': [], 'img2': []}
    diff_pairs = {'img1': [], 'img2': []}
    all_cls_imgs = {}

    for cl in cls:
        all_cls_imgs[cl] = get_cl_imgs(cl, classes_dir)

    for i in tqdm.tqdm(range(num_pairs)):
        # same:
        chosen_cl = npr.choice([k for k in all_cls_imgs.keys()], 1, replace=False)[0]
        chosen_imgs_idx = npr.choice([i for i in range(len(all_cls_imgs[chosen_cl]))], 2, replace=False)
        im1 = all_cls_imgs[chosen_cl][chosen_imgs_idx[0]]
        im2 = all_cls_imgs[chosen_cl][chosen_imgs_idx[1]]
        same_pairs['img1'].append(im1)
        same_pairs['img2'].append(im2)
        if chosen_imgs_idx[0] > chosen_imgs_idx[1]:
            del all_cls_imgs[chosen_cl][chosen_imgs_idx[0]]
            del all_cls_imgs[chosen_cl][chosen_imgs_idx[1]]
        else:
            del all_cls_imgs[chosen_cl][chosen_imgs_idx[1]]
            del all_cls_imgs[chosen_cl][chosen_imgs_idx[0]]

        if len(all_cls_imgs[chosen_cl]) == 0:
            del all_cls_imgs[chosen_cl]

        # diff

        chosen_cls = npr.choice([k for k in all_cls_imgs.keys()], 2, replace=False)
        im1_idx = npr.choice([i for i in range(len(all_cls_imgs[chosen_cls[0]]))], 1, replace=False)[0]
        im2_idx = npr.choice([i for i in range(len(all_cls_imgs[chosen_cls[1]]))], 1, replace=False)[0]
        im1 = all_cls_imgs[chosen_cls[0]][im1_idx]
        im2 = all_cls_imgs[chosen_cls[1]][im2_idx]
        diff_pairs['img1'].append(im1)
        diff_pairs['img2'].append(im2)
        del all_cls_imgs[chosen_cls[0]][im1_idx]
        del all_cls_imgs[chosen_cls[1]][im2_idx]

        if len(all_cls_imgs[chosen_cls[0]]) <= 1:
            del all_cls_imgs[chosen_cls[0]]
        if len(all_cls_imgs[chosen_cls[1]]) <= 1:
            del all_cls_imgs[chosen_cls[1]]

    return pd.DataFrame(same_pairs), pd.DataFrame(diff_pairs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--classes_dir", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/processed_pretraining_dataset/phase_perc_size/pretraining_fixed_{'train': 0.7, 'val': 0.2, 'test': 0.1}/test/")
    parser.add_argument("--output_file", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/image_pairs_lists/mutualy_exclusive/pretraining")
    parser.add_argument("--num_pairs", type=int,
                        default=3000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    same_pairs, diff_pairs = generate_division_pairs(args.classes_dir, args.num_pairs)
    same_pairs.to_csv(args.output_file + "_same.txt", sep=' ', index=False, header=False)
    diff_pairs.to_csv(args.output_file + "_diff.txt", sep=' ', index=False, header=False)
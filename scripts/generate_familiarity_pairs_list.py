import os
import glob
import argparse
import numpy.random as npr
import pandas as pd


def globall(path):
    return glob.glob(os.path.join(path, '*'))


def relall(path_lst, rel):
    return [os.path.relpath(p, rel) for p in path_lst]


def get_cl_imgs(cl, root):
    return relall(globall(cl), root)


def generate_class_pairs(class_imgs):
    pairs = npr.choice(class_imgs, (12, 2), replace=False)
    return pairs


def all_imgs_cls(cls, root):
    cls_imgs = {}
    for cl in cls:
        cls_imgs[cl] = get_cl_imgs(cl, root)
    return cls_imgs


def choose_diff(cls_imgs):
    pairs = []
    for i in range(1200):
        if len(cls_imgs.keys()) >= 2:
            pair_cls = npr.choice([k for k in cls_imgs.keys()], 2, replace=False)
            im1_idx = npr.choice([i for i in range(len(cls_imgs[pair_cls[0]]))], 1)[0]
            im2_idx = npr.choice([i for i in range(len(cls_imgs[pair_cls[1]]))], 1)[0]
            pairs.append([cls_imgs[pair_cls[0]][im1_idx], cls_imgs[pair_cls[1]][im2_idx]])
            del cls_imgs[pair_cls[0]][im1_idx]
            del cls_imgs[pair_cls[1]][im2_idx]
            if len(cls_imgs[pair_cls[0]]) == 0:
                del cls_imgs[pair_cls[0]]
            if len(cls_imgs[pair_cls[1]]) == 0:
                del cls_imgs[pair_cls[1]]
    return pairs



def generate_division_pairs(div_path):
    cls = globall(div_path)
    cols = ['img1', 'img2']
    same_pairs = pd.DataFrame(columns=cols)
    diff_pairs = pd.DataFrame(columns=cols)
    all_cls_imgs = {}

    for cl in cls:
        cls_imgs = get_cl_imgs(cl, div_path)
        chosen = npr.choice(cls_imgs, (2, 24), replace = False)
        same_cls = chosen[0]
        diff_cls = chosen[1]
        all_cls_imgs[cl] = diff_cls.tolist()

        same_pairs = same_pairs.append(pd.DataFrame(generate_class_pairs(same_cls), columns=cols))
    diff_pairs = pd.DataFrame(choose_diff(all_cls_imgs), columns=cols)
    return same_pairs, diff_pairs


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--classes_dir", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/processed_finetuning_dataset/phase_perc_size/pretraining_fixed_C_{'train': 220, 'val': 50, 'test': 50}/test/")
    parser.add_argument("--output_file", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/image_pairs_lists/mutualy_exclusive/C")

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    same_pairs, diff_pairs = generate_division_pairs(args.classes_dir)
    same_pairs.to_csv(args.output_file + "_same.txt", sep=' ', index=False, header=False)
    diff_pairs.to_csv(args.output_file + "_diff.txt", sep=' ', index=False, header=False)
from glob import glob
from tqdm import tqdm
from typing import List, Dict

from os import path

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def create_positive_pairs(cls_imgs: List[str]) -> pd.DataFrame:
    """
    Create a list of pairs of images, for the SAME cls pairs
    """
    num_imgs = len(cls_imgs) // 2
    pairs = np.random.choice(cls_imgs, [num_imgs, 2], replace=False)
    pairs = pd.DataFrame(pairs, columns=['img1', 'img2'])
    return pairs


def create_negative_pairs(cls2imgs: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Create a list of pairs of images for the DIFF cls pairs
    """
    total_items = 0
    idx = []
    cols = []

    cls_idx = {}
    cls_cols = {}

    # First shuffle the images within in each class
    for cls in cls2imgs:
        np.random.shuffle(cls2imgs[cls])
        total_items += len(cls2imgs[cls])
        curr_idx = []
        curr_cols = []

        for i, img in enumerate(cls2imgs[cls]):
            if i % 2 == 0:
                # curr_idx.append(path.join(cls, img))
                curr_idx.append(img)
            else:
                # curr_cols.append(path.join(cls, img))
                curr_cols.append(img)
        
        cls_idx[cls] = curr_idx
        cls_cols[cls] = curr_cols

        idx = idx + curr_idx
        cols = cols + curr_cols

    connections = pd.DataFrame(np.ones([len(idx), len(cols)]), index=idx, columns=cols)

    start_idx = 0
    start_col = 0
    # make sure not to set a connection between an image and itself
    for cls in cls2imgs:
        height = len(cls_idx[cls])
        width = len(cls_cols[cls])
        end_idx = start_idx + height
        end_col = start_col + width

        # Zeros the within cls connections
        connections.iloc[start_idx:end_idx, start_col:end_col] = 0
        start_idx = end_idx
        start_col = end_col
    
    # Shuffle the rows
    connections = connections.sample(frac=1.0)

    # Calculate the pairs (max flow)
    graph = csr_matrix(connections)
    matches = maximum_bipartite_matching(graph, perm_type='column')

    # Get the actual names of files

    diff_pairs = {'img1': [], 'img2': []}
    for i, j in enumerate(matches):
        if j >= 0:
            diff_pairs['img1'].append(connections.index[i])
            diff_pairs['img2'].append(connections.columns[j])

    diff_pairs = pd.DataFrame(diff_pairs)
    return diff_pairs


def generate_all_pairs(cls2imgs: Dict[str, List[str]]) -> pd.DataFrame:
    same = []
    for cls in cls2imgs:
        same.append(create_positive_pairs(cls2imgs[cls]))
    same = pd.concat(same)
    same['same'] = 1
    print(len(same))

    diff = create_negative_pairs(cls2imgs)
    diff['same'] = 0
    print(len(diff))

    return pd.concat([same, diff])


def globall(pth: str) -> List[str]:
    return glob(path.join(pth, '*'))


def get_verification_pairs(ds_path: str) -> pd.DataFrame:
    cls = globall(ds_path)
    cls2imgs = {}
    for cl in cls:
        cls2imgs[cl] = [path.relpath(img, ds_path) for img in globall(cl)]

    return generate_all_pairs(cls2imgs)


def split_to_balanced_lists(all_verification_pairs: pd.DataFrame, n_batches: int, batch_size: int) -> List[pd.DataFrame]:
    # Divide to positive and negatives
    positive = all_verification_pairs[all_verification_pairs['same'] == 1]
    negative = all_verification_pairs[all_verification_pairs['same'] == 0]

    # Shuffle rows order
    positive = positive.sample(frac=1)
    negative = negative.sample(frac=1)

    positive = positive.iloc[:batch_size * n_batches]
    negative = negative.iloc[:batch_size * n_batches]

    # Split the positive samples to equal size batches
    split_pos = np.array_split(positive, n_batches)

    # Set the negative samples to be of equal size to the positives, and split
    split_neg = np.array_split(negative, n_batches)

    # Concat the positives and negatives, and create the separate lists
    verification_tests = []
    for i in range(n_batches):
        print(i, len(split_pos[i]), len(split_neg[i]))
        verification_tests.append(pd.concat([split_pos[i], split_neg[i]]))

    return verification_tests


def process_ds(ds_path: str, output_dir: str, num_batches: int, batch_size: int) -> None:
    pairs = get_verification_pairs(ds_path)
    pairs.to_csv(path.join(output_dir, 'all.csv'))

    batches = split_to_balanced_lists(pairs, num_batches, batch_size)
    for i in range(num_batches):
        curr_batch = batches[i]
        curr_batch[curr_batch['same'] == 1][['img1', 'img2']].to_csv(path.join(output_dir, f'same_{i+1}.csv'), index=False, sep=' ', header=False)
        curr_batch[curr_batch['same'] == 0][['img1', 'img2']].to_csv(path.join(output_dir, f'diff_{i+1}.csv'), index=False, sep=' ', header=False)


if __name__ == '__main__':
    datasets = ['/home/ssd_storage/datasets/processed/verification_datasets/bird_species',
                "/home/ssd_storage/datasets/processed/phase_perc_size/individual_birds_single_species_{'train': 0.8, 'val': 0.2}/val",
                "/home/ssd_storage/datasets/processed/30_max_imgs_vggface2_mtcnn white_list_{'train': 0.8, 'val': 0.2}/val",
                "/home/ssd_storage/datasets/processed/num_classes/30_cls_inanimate_imagenet/val"]
    inner_dir = ['species', 'sociable_weavers', 'faces', 'inanimate_objects']

    pairs_output_dir = "/home/ssd_storage/experiments/Expertise/verification_pairs_lists/final_form"

    for ds, inner in zip(datasets, inner_dir):
        process_ds(ds, path.join(pairs_output_dir, inner), 30, 50)

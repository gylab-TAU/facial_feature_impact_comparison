from typing import List, Tuple

import pandas as pd
import numpy as np


def load_rdm(loc: str) -> pd.DataFrame:
    """Given a path to a csv RDM, loads it"""
    return pd.read_csv(loc, index_col=0)


def remove_rdm_redundancies(rdm:pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records from the RDM (choose only upper triangle)
    Remove distances from an item to itself (remove the main diagonal)
    """
    temp = rdm.copy()
    lt = np.tril(np.ones(rdm.shape), -1).astype(np.bool)

    temp = temp.where(lt == False, np.nan)
    np.fill_diagonal(temp.values, np.nan)
    return temp


def rdm_to_dist_list(rdm: pd.DataFrame) -> pd.DataFrame:
    """
    Given a full RDM, remove redundancies and return it in a format of a distance list
    """
    rdm = remove_rdm_redundancies(rdm)
    rdm.index = [str(i // 10) + ':' + str(col) for i, col in enumerate(rdm.index)]
    rdm.columns = [str(i // 10) + ':' + str(col) for i, col in enumerate(rdm.columns)]
    stacked = rdm.stack()
    no_diag = pd.DataFrame(stacked.dropna()).rename(columns={0: 'cos'})

    same = []
    for idx in no_diag.index:
        same.append(idx[0].split(':')[0] == idx[1].split(':')[0])
    no_diag['same'] = same
    return no_diag


def sample_dist_list(dist_list: pd.DataFrame, n_batches: int = 30) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample n_batches of equal size from the positive distances, and the negative distances
    All negative samples should have the same size as the positive samples
    """
    positive = dist_list[dist_list['same']]
    negative = dist_list[~dist_list['same']]

    positive_shuffled = positive.sample(frac=1)
    negative_shuffled = negative.sample(frac=1)

    split_pos  = np.array_split(positive_shuffled, n_batches)
    n_rows = len(split_pos[0])
    sampled_neg = negative_shuffled.sample(n_rows * n_batches)
    split_neg = np.array_split(sampled_neg, n_batches)

    for i in range(n_batches):
        split_pos[i] = split_pos[i]['cos'].to_numpy()
        split_neg[i] = split_neg[i]['cos'].to_numpy()

    return split_pos, split_neg


def balance_dist_list(dist_list: pd.DataFrame) -> pd.DataFrame:
    """
    Get all of the positive samples, and an equal number of negative_samples
    """
    positive = dist_list[dist_list['same']]
    negative = dist_list[~dist_list['same']]

    negative = negative.sample(len(positive))

    return pd.concat((positive, negative))


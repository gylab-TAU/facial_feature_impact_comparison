from typing import List, Tuple

from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score


def calc_graph_measurements(df: pd.DataFrame, label_col: str, value_col: str):
    """
    Given a DF with a label column {0,1} (assumes 0 is same because we usually work with distances)
    and a value column (floats)
    returns the TPR, FPR, thresholds, and AUROC
    """
    fpr, tpr, thresh = roc_curve(df[label_col], df[value_col], pos_label=0)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresh, roc_auc


def get_rdm_aucs(positives: List[np.ndarray], negatives: List[np.ndarray], n_batches: int):
    aucs = []

    for i in range(n_batches):
        # Mark labels
        pos_labels = np.ones_like(negatives[i])
        neg_labels = np.zeros_like(positives[i])
        labels = np.concatenate([neg_labels, pos_labels])

        # concatenate dists
        dists = np.concatenate([positives[i], negatives[i]])

        # Calculate AUCs
        fpr, tpr, thresholds = metrics.roc_curve(labels, dists, pos_label=1)
        aucs.append(metrics.auc(fpr, tpr))

    return aucs


def auc_stats(aucs: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    mean = np.mean(aucs)
    n = len(aucs)
    se = stats.sem(aucs)
    h = se * stats.t.ppf((1+confidence) / 2., n-1)
    return mean, h
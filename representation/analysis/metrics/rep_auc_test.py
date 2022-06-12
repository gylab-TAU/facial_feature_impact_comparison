from typing import List
import numpy as np
from sklearn import metrics
import torch


class RepAUCTester(object):
    def calc_performance(self, y_scores: torch.Tensor, y_true: torch.Tensor) -> List[float]:
        y_scores = y_scores.clone().detach().cpu().numpy()
        y_true = y_true.clone().detach().cpu().numpy()
        print(np.sum(y_true))
        print(y_true)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        return [auc]

    def get_metric_name(self) -> List[str]:
        return ['auc']

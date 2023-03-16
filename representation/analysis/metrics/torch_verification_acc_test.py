from typing import List
import torch


class VerificationAccTest(object):
    def calc_performance(self, y_scores: torch.Tensor, y_true: torch.Tensor) -> List[float]:
        with torch.no_grad():
            # Take same \ diff
            pos_dists = y_scores[y_true == 1]
            neg_dists = y_scores[y_true == 0]

            # Since we work with distances, mark same for all distances who are BELOW a threshold
            pos_thresh = (pos_dists <= y_scores[:, None]).float()
            neg_thresh = (neg_dists > y_scores[:, None]).float()
            all_thresh = torch.cat((pos_thresh, neg_thresh), dim=1)
            accuracies = all_thresh.mean(dim=1)
            best_thresh_idx = torch.argmax(accuracies)
            best_acc = accuracies[best_thresh_idx]
            best_thresh = pos_dists[best_thresh_idx]
        return [best_acc.item(), best_thresh.item()]

    def get_metric_name(self) -> List[str]:
        return ['best verification acc', 'threshold']

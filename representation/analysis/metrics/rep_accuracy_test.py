import numpy as np


class RepAccuracyTester(object):
    def __init__(self, threshold_matching):
        self.__threshold_matching = threshold_matching

    def calc_performance(self, y_scores, y_true):
        y_scores = np.asarray(y_scores)
        y_true = np.asarray(y_true)
        best_acc = 0
        best_threshold = 0
        for i in range(len(y_scores)):
            threshold = y_scores[i]
            y_test = self.__threshold_matching.get_matching(y_scores, threshold)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        return best_acc, best_threshold

    def get_metric_name(self) -> []:
        return ['acc@1', 'threshold']

import numpy as np


class NormalizedRepAverage(object):
    def calc_performance(self, y_scores, y_true):
        return np.mean(y_scores / np.max(y_scores))

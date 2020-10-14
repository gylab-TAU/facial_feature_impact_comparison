import numpy as np


class RepAverage(object):
    def calc_performance(self, y_scores, y_true):
        return np.mean(y_scores)

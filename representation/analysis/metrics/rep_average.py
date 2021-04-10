import numpy as np


class RepAverage(object):
    def calc_performance(self, y_scores, y_true):
        return np.mean(y_scores)


    def get_metric_name(self):
        return ["scores' mean"]

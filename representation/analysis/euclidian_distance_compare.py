import numpy as np


class EuclidianDistanceCompare(object):
    def compare(self, rep1: np.ndarray, rep2: np.ndarray):
        return np.linalg.norm(rep1 - rep2)
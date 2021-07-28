import numpy as np


class NormalizedCountCorrelatedFiring(object):
    def compare(self, rep1: np.ndarray, rep2: np.ndarray):
        bool1 = rep1.flatten() > 0
        bool2 = rep2.flatten() > 0
        return np.sum(bool1 * bool2) / np.sum(bool1 + bool2)

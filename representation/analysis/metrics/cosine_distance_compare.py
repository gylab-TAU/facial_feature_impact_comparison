import numpy as np
from scipy import spatial


class CosineDistanceCompare(object):
    def compare(self, rep1: np.ndarray, rep2: np.ndarray):
        return spatial.distance.cosine(rep1.flatten(), rep2.flatten())



class CosineThresholdMatching(object):
    def get_matching(self, scores, threshold):
        return scores <= threshold

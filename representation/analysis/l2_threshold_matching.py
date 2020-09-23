

class L2ThresholdMatching(object):
    def get_matching(self, scores, threshold):
        return scores <= threshold

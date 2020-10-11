import torch

class DatapointsRepComparer(object):
    def __init__(self, representation_extractor, comparison):
        self.__representation_extractor = representation_extractor
        self.__comparison = comparison

    def compare_datapoints(self, point1_key, point2_key, point1: torch.Tensor = None, point2: torch.Tensor = None):
        point1_rep = self.__representation_extractor.get_layers_representation(point1, point1_key)
        point2_rep = self.__representation_extractor.get_layers_representation(point2, point2_key)

        comparisons = {}
        # point1
        for key in point1_rep:
            comparisons[key] = self.__comparison.compare(rep1=point1_rep[key].cpu().numpy(), rep2=point2_rep[key].cpu().numpy())

        return comparisons
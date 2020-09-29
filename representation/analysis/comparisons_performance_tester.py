import numpy as np


class InputComparisonsPerformanceTester(object):
    def __init__(self, representation_performance_analyzer):
        self.__rep_perf_analyzer = representation_performance_analyzer

    def test_performance(self, comparisons_by_layers, labels_list):
        #  per layer we calculate the layer's accuracies
        reduced_performance = {}

        for key in comparisons_by_layers:
            reduced_performance[key] = self.__rep_perf_analyzer.calc_performance(comparisons_by_layers[key], labels_list)

        return reduced_performance

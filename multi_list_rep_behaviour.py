import numpy as np


class MultiListRepresentationBehaviour(object):
    def __init__(self, pairs_types_to_lists: dict, pairs_list_comparison, performance_tester, pairs_types_to_dir: dict, labels_dict=None):
        self.__pairs_types_to_lists = pairs_types_to_lists
        self.__labels_dict = labels_dict
        self.__pairs_types_to_dir = pairs_types_to_dir
        self.__performance_tester = performance_tester
        self.__pairs_list_comparison = pairs_list_comparison

    def test_behaviour(self, model):
        type_to_perf = {}
        type_to_comparisons = {}
        overall_max = 0
        for pairs_type in self.__pairs_types_to_lists:
            type_to_comparisons[pairs_type] = self.__pairs_list_comparison.compare_pairs(model,
                                                                                         self.__pairs_types_to_dir[pairs_type],
                                                                                         self.__pairs_types_to_lists[pairs_type],
                                                                                         pairs_type)
            if self.__labels_dict is not None:
                labels = self.__labels_dict[pairs_type]
            overall_max = max(np.max(type_to_comparisons[pairs_type]), overall_max)

            type_to_perf[pairs_type] = self.__performance_tester.test_performance(type_to_comparisons[pairs_type],
                                                                                  labels)
        return self.__normalize(type_to_perf, overall_max)

    def __normalize(self, type_to_perf: dict, overall_max: float):
        for pairs_type in type_to_perf:
            for layer in type_to_perf[pairs_type]:
                type_to_perf[pairs_type][layer] = type_to_perf[pairs_type][layer]/overall_max
        return type_to_perf

import numpy as np


class MultiListRepresentationBehaviour(object):
    def __init__(self, pairs_types_to_lists: dict, pairs_list_comparison, performance_tester, pairs_types_to_dir: dict):
        self.__pairs_types_to_lists = pairs_types_to_lists
        self.__pairs_types_to_dir = pairs_types_to_dir
        self.__performance_tester = performance_tester
        self.__pairs_list_comparison = pairs_list_comparison

    def test_behaviour(self, model):
        type_to_perf = {}
        type_to_comparisons = {}
        for pairs_type in self.__pairs_types_to_lists:
            type_to_comparisons[pairs_type], _ = self.__pairs_list_comparison.compare_pairs(model,
                                                                                         self.__pairs_types_to_dir[pairs_type],
                                                                                         self.__pairs_types_to_lists[pairs_type],
                                                                                         pairs_type)

            type_to_perf[pairs_type] = self.__performance_tester.test_performance(type_to_comparisons[pairs_type], None)
        return type_to_perf

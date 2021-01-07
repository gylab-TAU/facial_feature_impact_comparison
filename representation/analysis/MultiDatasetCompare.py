
import os

class MultiDatasetComparer(object):
    def __init__(self, pairs_types_to_dir, pairs_list_comparison, output_dir):
        self.__pairs_types_to_dir = pairs_types_to_dir
        self.__pairs_list_comparison = pairs_list_comparison
        self.__output_dir = output_dir

    def compare_lists(self, model):
        for pairs_type in self.__pairs_types_to_dir:
            layers_matrices = self.__pairs_list_comparison.compare_pairs(model,
                                                                 self.__pairs_types_to_dir[pairs_type],
                                                                 pairs_type)
            for l in layers_matrices:
                dir = os.path.join(self.__output_dir, pairs_type)
                os.makedirs(dir, exist_ok=True)
                layers_matrices[l].to_csv(os.path.join(dir, l + ".csv"))

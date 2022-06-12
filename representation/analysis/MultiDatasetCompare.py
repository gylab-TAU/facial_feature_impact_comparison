import mlflow
import os

class MultiDatasetComparer(object):
    def __init__(self, pairs_types_to_dir, pairs_list_comparison, output_dir):
        self.__pairs_types_to_dir = pairs_types_to_dir
        self.__pairs_list_comparison = pairs_list_comparison
        self.__output_dir = output_dir

    def compare_lists(self, model):
        print('self.__pairs_types_to_dir: ',self.__pairs_types_to_dir)
        for pairs_type in self.__pairs_types_to_dir:
            print("pairs_type: ", pairs_type)
            layers_matrices = self.__pairs_list_comparison.compare_pairs(model,
                                                                 self.__pairs_types_to_dir[pairs_type],
                                                                 pairs_type)
            print("layers_matrices: ", layers_matrices)
            #if we want only fc7 layer:
            for l in layers_matrices:
                print("l in layers_matrices: ", l)
                dir = os.path.join(self.__output_dir, pairs_type)
                print("dir: ", dir)
                print("self.__output_dir: ", self.__output_dir)
                print("pairs_type: ", pairs_type)
                os.makedirs(dir, exist_ok=True)
                rdm_path = os.path.join(dir, f'{pairs_type}_{l}.csv')
                print("layers_matrices saved to: ",rdm_path)
                layers_matrices[l].to_csv(rdm_path)
                mlflow.log_artifact(rdm_path)

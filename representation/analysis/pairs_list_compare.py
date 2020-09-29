from representation.analysis.datapoints_rep_compare import DatapointsRepComparer
from representation.acquisition.representation_save_hook import FileSystemHook
from representation.acquisition.representation_extraction import RepresentationExtractor
from representation.acquisition.rep_layer_model_dict import get_model_layers_dict
import os.path
from tqdm import tqdm


class PairsListComparison(object):
    def __init__(self, reps_cache_path, image_loader, comparison_calc):
        self.__reps_cache_path = reps_cache_path
        self.__image_loader = image_loader
        self.__comparison_calc = comparison_calc

    def compare_pairs(self, model, dataset_dir, pairs_list, progress_label):
        re = RepresentationExtractor(model,
                                     get_model_layers_dict(model),
                                     FileSystemHook(get_model_layers_dict(model), self.__reps_cache_path))

        comparisons_by_layers = {}

        for i in tqdm(range(pairs_list), desc=progress_label):
            im1_path = pairs_list[i][0]
            im2_path = pairs_list[i][1]
            im1_path = os.path.join(dataset_dir, im1_path)
            im2_path = os.path.join(dataset_dir, im2_path)

            im1 = self.__image_loader.load_image(im1_path)
            im2 = self.__image_loader.load_image(im2_path)

            im1_key = os.path.basename(im1_path)
            im2_key = os.path.basename(im2_path)

            comp = DatapointsRepComparer(representation_extractor=re, comparison=self.__comparison_calc)
            comparison = comp.compare_datapoints(im1_key, im2_key, im1, im2)

            for key in comparison:
                if key not in comparisons_by_layers:
                    comparisons_by_layers[key] = []
                comparisons_by_layers[key].append(comparison[key])

        return comparisons_by_layers

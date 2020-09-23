from representation.analysis.datapoints_rep_compare import DatapointsRepComparer

from representation.representation_save_hook import FileSystemHook
from representation.representation_extraction import RepresentationExtractor
from representation.rep_layer_model_dict import get_model_layers_dict
import os.path


class LFWTester(object):
    def __init__(self, labeled_pairs_list, reps_cache_path, image_loader, rep_accuracy_tester, comparison_calc):
        self.__labeled_pairs_list = labeled_pairs_list
        self.__reps_cache_path = reps_cache_path
        self.__image_loader = image_loader
        self.__rep_accuracy_tester = rep_accuracy_tester
        self.__comparison_calc = comparison_calc

    def test_lfw(self, model):
        # First we get all pair comparisons with the list of labels
        comparisons_by_layers, labels_list = self.__get_comparison_bey_layers(model)

        # Then, per layer we calculate the layer's accuracies
        best_threshold_accuracies = {}
        for key in comparisons_by_layers:
            best_threshold_accuracies[key] = self.__rep_accuracy_tester.calc_accuracy(comparisons_by_layers[key], labels_list)

        return best_threshold_accuracies

    def __get_comparison_bey_layers(self, model):
        labels_list = []

        re = RepresentationExtractor(model,
                                     get_model_layers_dict(model),
                                     FileSystemHook(get_model_layers_dict(model), self.__reps_cache_path))

        comparisons_by_layers = {}

        for im1_path, im2_path, label in self.__labeled_pairs_list:
            labels_list.append(label)

            im1 = self.__image_loader.load_image(im1_path)
            im2 = self.__image_loader.load_image(im2_path)

            im1_key = os.path.basename(im1_path)
            im2_key = os.path.basename(im2_path)

            comp = DatapointsRepComparer(representation_extractor=re, comparison=self.__comparison_calc)
            comparison = comp.compare_datapoints(im1_key, im2_key, im1, im2)

            for key in comparison:
                if key not in comparisons_by_layers:
                    comparisons_by_layers[key] = []
                comparisons_by_layers.append(comparison[key])

        return comparisons_by_layers, labels_list
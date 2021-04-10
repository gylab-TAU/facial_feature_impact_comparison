import lfw_test
import os
from representation.analysis.metrics.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.metrics.cosine_distance_compare import CosineDistanceCompare
from representation.analysis.pairs_list_compare import PairsListComparer
from representation.analysis.metrics.rep_accuracy_test import RepAccuracyTester
from representation.analysis.metrics.l2_threshold_matching import L2ThresholdMatching
from representation.analysis.metrics.cosine_threshold_matching import CosineThresholdMatching
from representation.analysis.performance_tester import PerformanceTester
from representation.acquisition.model_layer_dicts.rep_layer_model_dict import get_model_layers_dict
from representation.acquisition.model_layer_dicts.reflection_factory import ReflectionFactory


def get_lfw_test(config, image_loader):
    if 'LFW_TEST' not in config:
        return
    labeled_pairs_path = config['LFW_TEST']['labeled_pairs_path']
    pairs_list = []
    labels_list = []
    with open(labeled_pairs_path, 'r') as f:
        for line in f:
            labeled_pair = line.split(' ')
            labeled_pair[2] = int(labeled_pair[2].replace(os.linesep, ''))
            pairs_list.append(labeled_pair[:2])
            labels_list.append(labeled_pair[2])

    reps_cache_path = config['LFW_TEST']['reps_cache_path']

    if config['LFW_TEST']['comparison_metric'] == 'l2' or config['LFW_TEST']['comparison_metric'] == 'euclidian' :
        comparison_calc = EuclidianDistanceCompare()
        threshold_matcher = L2ThresholdMatching()
    if config['LFW_TEST']['comparison_metric'] == 'cos' or config['LFW_TEST']['comparison_metric'] =='cosine':
        comparison_calc = CosineDistanceCompare()
        threshold_matcher = CosineThresholdMatching()

    if 'reps_layers' in config['LFW_TEST']:
        layers_extractor = ReflectionFactory().get_dict_extractor(config['LFW_TEST']['reps_layers'])
    else:
        layers_extractor = get_model_layers_dict

    pairs_list_comparison = PairsListComparer(reps_cache_path, image_loader, comparison_calc, layers_extractor)
    performance_tester = PerformanceTester(RepAccuracyTester(threshold_matcher), pairs_list_comparison)

    return lfw_test.LFWTester(pairs_list, labels_list, config['LFW_TEST']['lfw_dir'], 'lfw', performance_tester)

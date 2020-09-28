import lfw_test
import os
from representation.analysis.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.rep_accuracy_test import RepAccuracyTester
from representation.analysis.l2_threshold_matching import L2ThresholdMatching


def get_lfw_test(config, image_loader):
    if 'LFW_TEST' not in config:
        return
    labeled_pairs_path = config['LFW_TEST']['labeled_pairs_path']
    labeled_pairs_list = []
    with open(labeled_pairs_path, 'r') as f:
        for line in f:
            labeled_pair = line.split(' ')
            labeled_pair[2] = int(labeled_pair[2].replace(os.linesep, ''))
            labeled_pairs_list.append(labeled_pair)

    reps_cache_path = config['LFW_TEST']['reps_cache_path']

    if config['LFW_TEST']['comparison_metric'] == 'l2' or config['LFW_TEST']['comparison_metric'] == 'euclidian' :
        comparison_calc = EuclidianDistanceCompare()
        threshold_matcher = L2ThresholdMatching()

    return lfw_test.LFWTester(labeled_pairs_list, reps_cache_path, image_loader, RepAccuracyTester(threshold_matcher),  comparison_calc, config['LFW_TEST']['lfw_dir'])
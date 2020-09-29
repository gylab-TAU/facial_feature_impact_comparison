import lfw_test
import os
from representation.analysis.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.pairs_list_compare import PairsListComparison
from representation.analysis.rep_accuracy_test import RepAccuracyTester
from representation.analysis.l2_threshold_matching import L2ThresholdMatching
from representation.analysis.performance_tester import PerformanceTester


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

    pairs_list_comparison = PairsListComparison(reps_cache_path, image_loader, comparison_calc)
    performance_tester = PerformanceTester(RepAccuracyTester(threshold_matcher), pairs_list_comparison)

    return lfw_test.LFWTester(pairs_list, labels_list, config['LFW_TEST']['lfw_dir'], 'lfw', performance_tester)

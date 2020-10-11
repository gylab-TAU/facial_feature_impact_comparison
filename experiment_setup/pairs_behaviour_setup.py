import json
import os
from representation.analysis.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.pairs_list_compare import PairsListComparison
from representation.analysis.comparisons_performance_tester import InputComparisonsPerformanceTester
from multi_list_rep_behaviour import MultiListRepresentationBehaviour
from representation.analysis.rep_average import RepAverage
from representation.acquisition.raw_model_layers_dict import get_model_layers_dict


def setup_pairs_reps_behaviour(config, image_loader):
    if 'REP_BEHAVIOUR' not in config:
        return
    pairs_paths = json.loads(config['REP_BEHAVIOUR']['pairs_paths'])
    pairs_types_to_lists = {}
    for pairs_type in pairs_paths:
        pairs_types_to_lists[pairs_type] = []
        with open(pairs_paths[pairs_type], 'r') as f:
            for line in f:
                labeled_pair = line.split(' ')
                labeled_pair[1] = labeled_pair[1].replace(os.linesep, '')
                pairs_types_to_lists[pairs_type].append(labeled_pair)

    pairs_image_dirs = json.loads(config['REP_BEHAVIOUR']['pairs_image_dirs'])

    reps_cache_path = config['REP_BEHAVIOUR']['reps_cache_path']

    if config['REP_BEHAVIOUR']['comparison_metric'] == 'l2' or config['REP_BEHAVIOUR']['comparison_metric'] == 'euclidian':
        comparison_calc = EuclidianDistanceCompare()

    pairs_list_comparison = PairsListComparison(reps_cache_path, image_loader, comparison_calc, get_model_layers_dict)
    performance_tester = InputComparisonsPerformanceTester(RepAverage())

    return MultiListRepresentationBehaviour(pairs_types_to_lists, pairs_list_comparison, performance_tester, pairs_image_dirs)

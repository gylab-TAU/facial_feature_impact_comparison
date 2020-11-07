import json
import os
from representation.analysis.metrics.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.layers_reduced_pairs_list_compare import LayersReducedPairsListComparer
from representation.analysis.input_comparisons_performance_tester import InputComparisonsPerformanceTester
from multi_list_rep_behaviour import MultiListRepresentationBehaviour
from representation.analysis.metrics.normalized_rep_average import NormalizedRepAverage
from representation.analysis.metrics.pref_test_stub import PerformanceTestStub
# from representation.acquisition.model_layer_dicts.fc_conv_layers_model_dict import get_model_layers_dict
from representation.analysis.pairs_list_compare import PairsListComparer
from representation.acquisition.model_layer_dicts.blauch_equivalent_list_model_dict import get_model_layers_dict
# from representation.acquisition.raw_model_layers_dict import get_model_layers_dict

def setup_pairs_reps_behaviour(config, image_loader):
    if 'REP_BEHAVIOUR' not in config:
        return
    pairs_paths = json.loads(config['REP_BEHAVIOUR']['pairs_paths'])
    pairs_types_to_lists = {}
    for pairs_type in pairs_paths:
        print(pairs_type)
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

    if config['REP_BEHAVIOUR']['reduce_performance'] == 'True':
        pairs_list_comparison = LayersReducedPairsListComparer(reps_cache_path, image_loader, comparison_calc, get_model_layers_dict)
        performance_tester = InputComparisonsPerformanceTester(NormalizedRepAverage())
    else:
        pairs_list_comparison = PairsListComparer(reps_cache_path, image_loader, comparison_calc, get_model_layers_dict)
        performance_tester = InputComparisonsPerformanceTester(PerformanceTestStub())

    return MultiListRepresentationBehaviour(pairs_types_to_lists, pairs_list_comparison, performance_tester, pairs_image_dirs)

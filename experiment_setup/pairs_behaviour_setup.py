import json
import os
import torch.utils.data as data

from representation.analysis.MultiDatasetCompare import MultiDatasetComparer
from representation.analysis.metrics.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.metrics.cosine_distance_compare import CosineDistanceCompare
from representation.analysis.pairs_list_compare import PairsListComparer
from representation.acquisition.model_layer_dicts.blauch_equivalent_list_model_dict import get_model_layers_dict
from representation.analysis.multi_list_comparer import MultiListComparer
from representation.analysis.rep_dist_mat import DistMatrixComparer
from representation.activations.activation_acquisition import ActivationAcquisition


def setup_pairs_reps_behaviour(config, image_loader):
    if 'REP_BEHAVIOUR' not in config:
        return

    if 'activations' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['activations'] == 'True':
        ds_path = config['REP_BEHAVIOUR']['activations_dataset']
        whitelist = json.loads(config['REP_BEHAVIOUR']['whitelist'])
        activations_dataset = data.DataLoader(
            image_loader.load_dataset(ds_path),
            batch_size=int(config['REP_BEHAVIOUR']['batch_size']),
            num_workers=int(config['REP_BEHAVIOUR']['workers']),
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        return ActivationAcquisition(activations_dataset, whitelist, int(config['MODELLING']['num_classes']))

    

    reps_cache_path = config['REP_BEHAVIOUR']['reps_cache_path']

    if config['REP_BEHAVIOUR']['comparison_metric'] == 'l2' or config['REP_BEHAVIOUR']['comparison_metric'] == 'euclidian':
        comparison_calc = EuclidianDistanceCompare()
    if config['REP_BEHAVIOUR']['comparison_metric'] == 'cos' or config['REP_BEHAVIOUR']['comparison_metric'] == 'CosineSimilarity':
        comparison_calc = CosineDistanceCompare()
    if 'dist_mat' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['dist_mat'] == 'True':
        return MultiDatasetComparer(json.loads(config['REP_BEHAVIOUR']['datasets']),
                                    DistMatrixComparer(reps_cache_path, image_loader, comparison_calc, get_model_layers_dict),
                                    config['REP_BEHAVIOUR']['reps_results_path'])

    else:
        pairs_image_dirs = json.loads(config['REP_BEHAVIOUR']['pairs_image_dirs'])
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
        pairs_list_comparison = PairsListComparer(reps_cache_path, image_loader, comparison_calc,
                                                  get_model_layers_dict)
        return MultiListComparer(pairs_types_to_lists, pairs_image_dirs, pairs_list_comparison)


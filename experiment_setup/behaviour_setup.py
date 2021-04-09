import json
import os
import torch.utils.data as data
import torch

from data_prep.Datasets.pairs_dataset import ComparisonDataset
from data_prep.transforms_config import get_transforms
from representation.analysis.MultiDatasetCompare import MultiDatasetComparer
from representation.analysis.pairs_list_compare_ref import PairsListComparer
from representation.acquisition.model_layer_dicts.blauch_equivalent_list_model_dict import get_model_layers_dict
from representation.analysis.multi_list_comparer_ref import MultiListComparer
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
            batch_size=int(config['MODELLING']['batch_size']),
            num_workers=int(config['MODELLING']['workers']),
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        return ActivationAcquisition(activations_dataset, whitelist, int(config['MODELLING']['num_classes']))

    reps_cache_path = config['REP_BEHAVIOUR']['reps_cache_path']

    if config['REP_BEHAVIOUR']['comparison_metric'] in torch.nn.__dict__:
        comparison_calc = torch.nn.__dict__[config['REP_BEHAVIOUR']['comparison_metric']]()
    if config['REP_BEHAVIOUR']['comparison_metric'] == 'l2' or config['REP_BEHAVIOUR']['comparison_metric'] == 'euclidian':
        comparison_calc = torch.nn.PairwiseDistance()
    if torch.cuda.is_available():
        comparison_calc.cuda()

    if 'dist_mat' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['dist_mat'] == 'True':
        return MultiDatasetComparer(json.loads(config['REP_BEHAVIOUR']['datasets']),
                                    DistMatrixComparer(reps_cache_path, image_loader, comparison_calc,
                                                       get_model_layers_dict),
                                    config['REP_BEHAVIOUR']['reps_results_path'])

    else:
        pairs_image_dirs = json.loads(config['REP_BEHAVIOUR']['pairs_image_dirs'])
        pairs_paths = json.loads(config['REP_BEHAVIOUR']['pairs_paths'])
        pairs_dataloaders = {}
        for key in pairs_image_dirs:
            pairs_dataloaders[key] = torch.utils.data.DataLoader(
                ComparisonDataset(pairs_image_dirs[key], pairs_paths[key], get_transforms(config['DATASET'])),
                batch_size=int(config['REP_BEHAVIOUR']['batch_size']),
                num_workers=int(config['REP_BEHAVIOUR']['workers']),
                shuffle=False,
                pin_memory=True,
                drop_last=False)
        pairs_list_comparison = PairsListComparer(reps_cache_path, comparison_calc, get_model_layers_dict)
        return MultiListComparer(pairs_dataloaders, pairs_list_comparison)


import json
import os
import torch.utils.data as data

from representation.analysis.MultiDatasetCompare import MultiDatasetComparer
from representation.analysis.metrics.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.metrics.cosine_distance_compare import CosineDistanceCompare
from representation.analysis.pairs_list_compare import PairsListComparer
from representation.analysis.multi_list_comparer import MultiListComparer
from representation.analysis.rep_dist_mat import DistMatrixComparer
from representation.analysis.metrics.correlated_firing import CountCorrelatedFiring
from representation.analysis.metrics.normalized_correlated_firing import NormalizedCountCorrelatedFiring
from representation.analysis.metrics.count_firing import CountFiring
from representation.activations.activation_acquisition import ActivationAcquisition
from representation.activations.identification_error_acquitision import IdentificationErrorAcquisition
from representation.activations.deep_layers_activations import DeepLayersActivations
from representation.activations.multi_list_activations_acquisition import MultiListAcquisition
from representation.acquisition.model_layer_dicts.reflection_factory import ReflectionFactory
from representation.activations.strongest_activating_image import StrongestActivatingImageRetrieval

from data_prep.Datasets.img_label_path_dataset import ImgLabelPathDataset
from torchvision import transforms


def setup_pairs_reps_behaviour(config, image_loader):
    if 'REP_BEHAVIOUR' not in config:
        return

    if 'strongest_activating_image' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['strongest_activating_image'] == 'True':
        dataset_dir = config['REP_BEHAVIOUR']['dataset_dir']

        center_crop_tt = transforms.Compose([
            transforms.Resize(json.loads(config['DATASET']['image_size'])),
            transforms.CenterCrop(int(config['DATASET']['post_crop_im_size'])),
            transforms.ToTensor(),
            transforms.Normalize(json.loads(config['DATASET']['dataset_means']),
                                 json.loads(config['DATASET']['dataset_stds']))
        ])

        dataset = data.DataLoader(
            ImgLabelPathDataset(dataset_dir, center_crop_tt),
            batch_size=int(config['MODELLING']['batch_size']),
            num_workers=int(config['MODELLING']['workers']),
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        return StrongestActivatingImageRetrieval(dataset)

    if 'identification_errors' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['identification_errors'] == 'True':
        ds_path = config['REP_BEHAVIOUR']['activations_dataset']
        activations_dataset = data.DataLoader(
            image_loader.load_dataset(ds_path, with_path=True),
            batch_size=1,
            num_workers=int(config['REP_BEHAVIOUR']['workers']),
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        return IdentificationErrorAcquisition(activations_dataset)

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
    rep_dict_factory = ReflectionFactory()
    get_model_layers_dict = rep_dict_factory.get_dict_extractor(config['REP_BEHAVIOUR']['reps_layers'])

    if 'deep_activations' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['deep_activations'] == 'True':
        imgs_dirs = json.loads(config['REP_BEHAVIOUR']['imgs_dirs'])
        imgs_paths = json.loads(config['REP_BEHAVIOUR']['imgs_paths'])
        imgs_types_to_lists = {}
        for imgs_type in imgs_paths:
            print(imgs_type)
            imgs_types_to_lists[imgs_type] = []
            with open(imgs_paths[imgs_type], 'r') as f:
                for line in f:
                    im = line.replace('\n', '')
                    imgs_types_to_lists[imgs_type].append(im)
        return MultiListAcquisition(imgs_types_to_lists, imgs_dirs, DeepLayersActivations(reps_cache_path, image_loader, get_model_layers_dict))



    if config['REP_BEHAVIOUR']['comparison_metric'] == 'l2' or config['REP_BEHAVIOUR']['comparison_metric'] == 'euclidian':
        comparison_calc = EuclidianDistanceCompare()
    if config['REP_BEHAVIOUR']['comparison_metric'] == 'cos' or config['REP_BEHAVIOUR']['comparison_metric'] == 'CosineSimilarity':
        comparison_calc = CosineDistanceCompare()
    if config['REP_BEHAVIOUR']['comparison_metric'] == 'correlated_firing_count':
        comparison_calc = CountCorrelatedFiring()
    if config['REP_BEHAVIOUR']['comparison_metric'] == 'normalized_correlated_firing_count':
        comparison_calc = NormalizedCountCorrelatedFiring()
    if config['REP_BEHAVIOUR']['comparison_metric'] == 'firing_count':
        comparison_calc = CountFiring()

    if 'dist_mat' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['dist_mat'] == 'True':
        return MultiDatasetComparer(json.loads(config['REP_BEHAVIOUR']['datasets']),
                                    DistMatrixComparer(reps_cache_path, image_loader, comparison_calc, ReflectionFactory().get_dict_extractor(config['REP_BEHAVIOUR']['reps_layers'])),
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
                    print(labeled_pair)
                    labeled_pair[1] = labeled_pair[1].replace(os.linesep, '')
                    pairs_types_to_lists[pairs_type].append(labeled_pair)
        pairs_list_comparison = PairsListComparer(reps_cache_path, image_loader, comparison_calc,
                                                  get_model_layers_dict)
        return MultiListComparer(pairs_types_to_lists, pairs_image_dirs, pairs_list_comparison)


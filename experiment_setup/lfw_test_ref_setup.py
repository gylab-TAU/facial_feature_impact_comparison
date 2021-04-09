import lfw_test
import torch
import os
from data_prep.Datasets.pairs_dataset import ComparisonDataset
from data_prep.transforms_config import get_transforms
from representation.analysis.metrics.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.metrics.cosine_distance_compare import CosineDistanceCompare
from representation.analysis.pairs_list_compare_ref import PairsListComparer
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
    images_path = config['LFW_TEST']['lfw_dir']
    pairs_list = []
    labels_list = []


    with open(labeled_pairs_path, 'r') as f:
        for line in f:
            labeled_pair = line.split(' ')
            labeled_pair[2] = int(labeled_pair[2].replace(os.linesep, ''))
            pairs_list.append(labeled_pair[:2])
            labels_list.append(labeled_pair[2])

    dl = torch.utils.data.DataLoader(
        ComparisonDataset(images_path, pairs_list, get_transforms(config['DATASET'])),
        batch_size=int(config['LFW_TEST']['batch_size']),
        num_workers=int(config['LFW_TEST']['workers']),
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    reps_cache_path = config['LFW_TEST']['reps_cache_path']

    if config['LFW_TEST']['comparison_metric'] == 'l2' or config['LFW_TEST']['comparison_metric'] == 'euclidian' :
        comparison_calc = torch.nn.PairwiseDistance()
        if torch.cuda.is_available():
            comparison_calc.cuda()
        threshold_matcher = L2ThresholdMatching()
    if config['LFW_TEST']['comparison_metric'] == 'cos' or config['LFW_TEST']['comparison_metric'] =='cosine':
        calc = torch.nn.CosineSimilarity()
        if torch.cuda.is_available():
            calc.cuda()
        comparison_calc = lambda x, y:  1 - calc(x,y)
        threshold_matcher = CosineThresholdMatching()

    if 'reps_layers' in config['LFW_TEST']:
        layers_extractor = ReflectionFactory().get_dict_extractor(config['LFW_TEST']['reps_layers'])
    else:
        layers_extractor = get_model_layers_dict

    pairs_list_comparison = PairsListComparer(reps_cache_path, comparison_calc, layers_extractor)
    performance_tester = PerformanceTester(RepAccuracyTester(threshold_matcher), pairs_list_comparison)

    return lfw_test.LFWTester(pairs_list, labels_list, , 'lfw', performance_tester)

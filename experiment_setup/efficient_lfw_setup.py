import numpy as np
import pandas as pd

from representation.analysis.metrics.rep_auc_test import RepAUCTester
from representation.analysis.metrics.torch_verification_acc_test import VerificationAccTest
from representation.acquisition.model_layer_dicts.rep_layer_model_dict import get_model_layers_dict
from representation.acquisition.model_layer_dicts.reflection_factory import ReflectionFactory
from data_prep.Datasets.verification_dataset import VerificationDataset
from representation.analysis.efficient_lfw import EfficientLFW

import torch.utils.data as data


def get_lfw_test(config, image_loader):
    if 'LFW_TEST' not in config:
        return
    labeled_pairs_path = config['LFW_TEST']['labeled_pairs_path']

    # Read the verification pairs txt file, and break to columns of im1, im2, same
    # and build the dataset
    df = pd.read_csv(labeled_pairs_path, sep=' ', index_col=False, names=['im1', 'im2', 'same'], dtype={'im1': str, 'im2': str, 'same': np.int32})
    pairs_list = df[['im1', 'im2']].to_records(index=False)
    labels_list = df['same'].to_list()
    dataset = VerificationDataset(config['LFW_TEST']['lfw_dir'], pairs_list, image_loader, labels_list)
    dl = data.DataLoader(
        dataset,
        batch_size=1, # 256
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        drop_last=False)

    reps_cache_path = config['LFW_TEST']['reps_cache_path']

    # Which layers to run the verification test on
    if 'reps_layers' in config['LFW_TEST']:
        layers_extractor = ReflectionFactory().get_dict_extractor(config['LFW_TEST']['reps_layers'])
    else:
        layers_extractor = get_model_layers_dict

    # Which summary metric to use for the verification test, defaults to best accuracy
    if 'summary_metric' in config['LFW_TEST']:
        if config['LFW_TEST']['summary_metric'] == 'best_acc':
            summary_metric = VerificationAccTest()
        elif config['LFW_TEST']['summary_metric'] == 'auc':
            summary_metric = RepAUCTester()
    else:
        summary_metric = VerificationAccTest()

    return EfficientLFW(reps_cache_path, dl, layers_extractor, summary_metric, 'Verification test', config['LFW_TEST'][
        'comparison_metric'])


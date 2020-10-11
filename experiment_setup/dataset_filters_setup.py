import json
import os
from data_prep.num_classes_processing import NumClassProcessor
from data_prep.phase_size_processing import PhaseSizeProcessor
from data_prep.multi_stage_processing import MultiStageProcessor


def setup_dataset_filter(config):
    filter_list = []
    filter_names_list = json.loads(config['DATASET']['filters'])
    for filter_name in filter_names_list:
        if filter_name in ['phase_size']:
            filter_list.append(phase_size_filter_setup(config))
        elif filter_name in ['class_num']:
            filter_list.append(class_num_filter_setup(config))

    return MultiStageProcessor(filter_list)


def phase_size_filter_setup(config):
    phase_size_dict = json.loads(config['DATASET']['phase_size_dict'])

    class_filter_dataset_dir = config['DATASET']['class_filter_dataset_dir']
    class_size_filter_output_dir = os.path.join(config['DATASET']['processed_dataset_root'],
                                                class_filter_dataset_dir)

    return PhaseSizeProcessor(class_size_filter_output_dir, phase_size_dict)


def class_num_filter_setup(config):
    class_num_filter_output_dir = os.path.join(config['DATASET']['processed_dataset_root'],
                                               config['DATASET']['dataset_size_filter_dir'])

    max_num_classes = int(config['DATASET']['max_num_classes'])
    min_num_classes = int(config['DATASET']['min_num_classes'])

    return NumClassProcessor(min_num_classes, max_num_classes, class_num_filter_output_dir)
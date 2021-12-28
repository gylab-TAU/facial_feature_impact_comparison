import json
import os
from data_prep.num_classes_processing import NumClassProcessor
from data_prep.phase_size_processing import PhaseSizeProcessor
from data_prep.phase_percentage_processing import PhasePercentageProcessor
from data_prep.multi_stage_processing import MultiStageProcessor
from data_prep.mtcnn_align_processing import MTCNNAlignProcessor
from data_prep.class_size_processing import ClassSizeProcessing


def setup_dataset_filter(config):
    filter_list = []
    filter_names_list = json.loads(config['DATASET']['filters'])
    for filter_name in filter_names_list:
        if filter_name in ['phase_size']:
            filter_list.append(phase_size_filter_setup(config))
        if filter_name in ['phase_perc_size']:
            filter_list.append(phase_perc_size_filter_setup(config))
        if filter_name in ['class_num']:
            filter_list.append(class_num_filter_setup(config))
        elif filter_name in ['class_size']:
            filter_list.append(class_size_filter_setup(config))


    return MultiStageProcessor(filter_list)


def phase_size_filter_setup(config):
    phase_size_dict = json.loads(config['DATASET']['phase_size_dict'])

    class_filter_dataset_dir = config['DATASET']['class_filter_dataset_dir']
    class_size_filter_output_dir = os.path.join(config['DATASET']['processed_dataset_root'],
                                                class_filter_dataset_dir)

    return PhaseSizeProcessor(class_size_filter_output_dir, phase_size_dict)


def phase_perc_size_filter_setup(config):
    phase_size_dict = json.loads(config['DATASET']['phase_size_dict'])

    class_filter_dataset_dir = config['DATASET']['class_filter_dataset_dir']
    class_size_filter_output_dir = os.path.join(config['DATASET']['processed_dataset_root'],
                                                class_filter_dataset_dir)

    return PhasePercentageProcessor(class_size_filter_output_dir, phase_size_dict)



def mtcnn_align_filter_setup(config):
    mtcnn_align_dim = config['DATASET']['mtcnn_align_image_dim']
    mtcnn_align_margin = config['DATASET']['mtcnn_align_margin']

    processed_dataset_root = config['DATASET']['processed_dataset_root']

    return MTCNNAlignProcessor(image_size=mtcnn_align_dim, margin=mtcnn_align_margin, output_dataset_dir=processed_dataset_root)


def class_num_filter_setup(config):
    class_num_filter_output_dir = os.path.join(config['DATASET']['processed_dataset_root'],
                                               config['DATASET']['dataset_size_filter_dir'])
    class_depth = int(config['DATASET']['class_depth'])
    max_num_classes = int(config['DATASET']['max_num_classes'])
    min_num_classes = int(config['DATASET']['min_num_classes'])

    return NumClassProcessor(min_num_classes, max_num_classes, class_num_filter_output_dir, depth=class_depth)

def class_size_filter_setup(config):
    class_num_filter_output_dir = os.path.join(config['DATASET']['processed_dataset_root'],
                                               config['DATASET']['class_size_filter_dir'])

    min_class_size = int(config['DATASET']['min_class_size'])

    return ClassSizeProcessing(min_class_size, class_num_filter_output_dir)
import os
import glob
from data_prep.util import transfer_datapoints
import numpy as np

class PreexistingPhaseSizeProcessor(object):
    """
    A setup object, taking a raw dataset and filtering it according to constant phase size
    """
    def __init__(self, output_dataset_dir: str, phase_size_dict: {}, data_name_filter='*', class_name_filter='*', leading_phase='train'):
        self.output_dataset_dir = output_dataset_dir
        self.phase_size_dict = phase_size_dict
        self.data_name_filter = data_name_filter
        self.class_name_filter = class_name_filter
        self.leading_phase = leading_phase

    def process_dataset(self, raw_dataset_dir, dataset_name):
        class_list = self.__select_classes(raw_dataset_dir)

        filtered_dataset_output = self.__get_output_path(dataset_name)

        if os.path.exists(filtered_dataset_output):
            return filtered_dataset_output, glob.glob(os.path.join(filtered_dataset_output, list(self.phase_size_dict.keys())[0], self.class_name_filter))

        num_classes_to_use = 0
        for phase in self.phase_size_dict:
            for cl in class_list:
                data_points = glob.glob(os.path.join(raw_dataset_dir, phase, cl, self.data_name_filter))
                dest_dir = os.path.join(filtered_dataset_output, phase)
                transfer_datapoints(dest_dir, raw_dataset_dir, data_points)

        return filtered_dataset_output, num_classes_to_use

    def __select_classes(self, raw_dataset_dir):
        class_filter = os.path.join(raw_dataset_dir, self.leading_phase, self.class_name_filter)
        class_list = glob.glob(class_filter)
        filtered_class_list = []
        for cl in class_list:
            if glob.glob(os.path.join(cl, '*')) >= self.phase_size_dict[self.leading_phase]:
                filtered_class_list.append(os.path.basename(cl))
        return filtered_class_list

    def __get_output_path(self, dataset_name):
        return os.path.join(self.output_dataset_dir, f'{dataset_name}_{str(self.phase_size_dict)}')

import os
import glob
from data_prep.util import transfer_datapoints
import numpy as np


class DatasetSizeFilter(object):
    """
    A setup object, taking a raw dataset and filtering it according to constant phase size,
    and a limited range of classes to use
    """
    def __init__(self, output_dataset_dir: str, phase_size_dict: {}, max_num_classes: int, min_num_classes : int = 2, data_name_filter='*', class_name_filter='*'):
        self.output_dataset_dir = output_dataset_dir
        self.phase_size_dict = phase_size_dict
        self.max_num_classes = max_num_classes
        self.min_num_classes = min_num_classes
        self.data_name_filter = data_name_filter
        self.class_name_filter = class_name_filter

    def process_dataset(self, raw_dataset_dir, dataset_name):
        class_filter = os.path.join(raw_dataset_dir, self.class_name_filter)
        class_list = glob.glob(class_filter)
        num_classes_to_use = self.max_num_classes

        filtered_dataset_output = os.path.join(self.output_dataset_dir, dataset_name)

        assert len(class_list) >= self.min_num_classes
        if self.max_num_classes > len(class_list) or self.max_num_classes == 0:
            num_classes_to_use = len(class_list)

        min_data_point = sum(self.phase_size_dict.values())

        for i in range(num_classes_to_use):
            class_name = os.path.basename(class_list[i])
            class_dir_path = class_list[i]
            data_points = glob.glob(os.path.join(class_dir_path, self.data_name_filter))
            num_datapoints = len(data_points)

            if num_datapoints >= min_data_point:
                reduced_data = data_points
                # for each phase we choose a specific amount of datapoint for every id, then remove the data points we
                #     use for selection of the next phase
                for phase in self.phase_size_dict.keys():
                    phase_data = np.random.choice(data_points, self.phase_size_dict[phase], replace=False)
                    transfer_datapoints(filtered_dataset_output, phase, class_name, phase_data)
                    reduced_data = np.setdiff1d(reduced_data, phase_data)

        return filtered_dataset_output


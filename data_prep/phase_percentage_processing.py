import os
import glob
from data_prep.util import transfer_datapoints
import numpy as np


class PhasePercentageProcessor(object):
    """
    A setup object, taking a raw dataset and filtering it according to a given percentage of overall datapoints per phase
    """
    def __init__(self, output_dataset_dir: str, phase_percentage_dict: {}, data_name_filter='*', class_name_filter='*'):
        self.output_dataset_dir = output_dataset_dir
        self.phase_percentage_dict = phase_percentage_dict
        self.data_name_filter = data_name_filter
        self.class_name_filter = class_name_filter

    def process_dataset(self, raw_dataset_dir, dataset_name):
        class_filter = os.path.join(raw_dataset_dir, self.class_name_filter)
        class_list = glob.glob(class_filter)

        filtered_dataset_output = self.__get_output_path(dataset_name)

        if os.path.exists(filtered_dataset_output):
            return filtered_dataset_output, len(glob.glob(os.path.join(filtered_dataset_output, list(self.phase_percentage_dict.keys())[0], self.class_name_filter)))

        max_classes = len(class_list)

        for i in range(max_classes):
            class_name = os.path.basename(class_list[i])
            class_dir_path = class_list[i]
            data_points = glob.glob(os.path.join(class_dir_path, self.data_name_filter))
            num_datapoints = len(data_points)
            reduced_data = data_points
            # for each phase we choose a specific amount of datapoint for every id, then remove the data points we
            #     use for selection of the next phase
            for phase in self.phase_percentage_dict.keys():
                phase_perc = self.phase_percentage_dict[phase]
                phase_size = int(phase_perc * num_datapoints)
                phase_data = np.random.choice(reduced_data, phase_size, replace=False)
                dest_dir = os.path.join(filtered_dataset_output, phase)
                transfer_datapoints(dest_dir, raw_dataset_dir, phase_data)
                reduced_data = np.setdiff1d(reduced_data, phase_data)

        num_classes_to_use = len(glob.glob(
            os.path.join(filtered_dataset_output, list(self.phase_percentage_dict.keys())[0], self.class_name_filter)))

        return filtered_dataset_output, num_classes_to_use

    def __get_output_path(self, dataset_name):
        return os.path.join(self.output_dataset_dir, f'{dataset_name}_{str(self.phase_percentage_dict)}')



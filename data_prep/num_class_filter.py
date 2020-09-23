import os
import glob
from data_prep.util import transfer_datapoints
import numpy as np


class NumClassFilter(object):
    def __init__(self, min_class, max_class, output_dataset_dir, class_name_filter=os.path.join('*', '*'), data_name_filter='*'):
        self.__min_num_class = min_class
        self.__max_num_class = max_class
        self.__class_name_filter = class_name_filter
        self.__output_dataset_dir = output_dataset_dir
        self.__data_name_filter = data_name_filter

    def process_dataset(self, raw_dataset_dir, dataset_name):
        class_filter = os.path.join(raw_dataset_dir, self.__class_name_filter)
        class_names = np.unique(np.apply_along_axis(lambda cl: os.path.basename(cl), 0, np.array(glob.glob(class_filter))))
        num_classes_to_use = self.__max_num_classes

        assert len(class_names) >= self.__min_num_classes
        if self.__max_num_classes > len(class_names) or self.__max_num_classes == 0:
            num_classes_to_use = class_names.shape[0]

        classes_to_use = np.random.choice(class_names, num_classes_to_use)

        filtered_dataset_output = os.path.join(self.__output_dataset_dir, f'{dataset_name}_num-classes_{num_classes_to_use}')

        for i in range(num_classes_to_use):
            class_dir_path = glob.glob(os.path.join(raw_dataset_dir, '*', classes_to_use[i]))
            data_points = glob.glob(os.path.join(class_dir_path, self.__data_name_filter))

            transfer_datapoints(filtered_dataset_output, raw_dataset_dir, data_points)

        return filtered_dataset_output, num_classes_to_use
import os
import glob
from typing import List
from sklearn.model_selection import KFold
from data_prep.util import transfer_datapoints
import numpy as np


class KFoldProcessor(object):
    """
    Splitting the dataset to k folds
    """
    def __init__(self, output_dataset_dir: str, k: int, data_name_filter='*', class_name_filter='*'):
        self.output_dataset_dir = output_dataset_dir
        assert(k > 1)
        self.k = k
        self.kf = KFold(n_splits=k)
        self.data_name_filter = data_name_filter
        self.class_name_filter = class_name_filter

    def process_dataset(self, raw_dataset_dir, dataset_name):
        print('Running phase percentage size filter for:')
        print(f'dir: {raw_dataset_dir}')
        class_filter = os.path.join(raw_dataset_dir, self.class_name_filter)
        class_list = glob.glob(class_filter)

        filtered_dataset_output = self.__get_output_path(dataset_name)

        if os.path.exists(filtered_dataset_output):
            return filtered_dataset_output, len(glob.glob(os.path.join(filtered_dataset_output, list(self.phase_percentage_dict.keys())[0], self.class_name_filter)))

        max_classes = len(class_list)
        folds = []
        for i in range(self.k):
            dest_dir = os.path.join(filtered_dataset_output, f'fold_{i}')
            folds.append(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)

        for i in range(max_classes):
            class_name = os.path.basename(class_list[i])
            class_dir_path = class_list[i]
            data_points = glob.glob(os.path.join(class_dir_path, self.data_name_filter))
            num_datapoints = len(data_points)
            self.__get_cls_kfolds(np.ndarray(data_points), filtered_dataset_output, raw_dataset_dir)

        num_classes_to_use = len(glob.glob(
            os.path.join(filtered_dataset_output, f'fold_0', 'train', self.class_name_filter)))

        return folds, num_classes_to_use

    def __get_cls_kfolds(self, datapoints: np.ndarray, filtered_dataset_output: str, raw_dataset_dir: str):
        for i, (train_idx, test_idx) in enumerate(self.kf.split(datapoints)):
            dest_dir = os.path.join(filtered_dataset_output, f'fold_{i}')
            transfer_datapoints(os.path.join(dest_dir, 'train'), raw_dataset_dir, datapoints[train_idx])
            transfer_datapoints(os.path.join(dest_dir, 'val'), raw_dataset_dir, datapoints[test_idx])

    def __get_output_path(self, dataset_name):
        return os.path.join(self.output_dataset_dir, f'{dataset_name}_{self.k}-folds')



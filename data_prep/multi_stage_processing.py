import os
import glob


class MultiStageProcessor(object):
    def __init__(self, ordered_filters):
        self.__filters = ordered_filters

    def process_dataset(self, raw_dataset_dir, dataset_name):
        curr_dataset_dir = raw_dataset_dir
        curr_dataset_name = dataset_name
        num_classes = self.__get_initial_num_classes(raw_dataset_dir)
        for filter in self.__filters:
            curr_dataset_dir, num_classes = filter.process_dataset(curr_dataset_dir, curr_dataset_name)
            curr_dataset_name = os.path.basename(curr_dataset_dir)

        return curr_dataset_dir, num_classes

    def __get_initial_num_classes(self, raw_dataset_dir):
        if os.path.join(raw_dataset_dir, 'train') not in glob.glob(os.path.join(raw_dataset_dir, '*')):
            return len(glob.glob(os.path.join(raw_dataset_dir, '*')))
        return len(glob.glob(os.path.join(raw_dataset_dir, 'train', '*')))
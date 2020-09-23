import os


class MultiStageFilter(object):
    def __init__(self, ordered_filters):
        self.__filters = ordered_filters

    def process_dataset(self, raw_dataset_dir, dataset_name):
        curr_dataset_dir = raw_dataset_dir
        curr_dataset_name = dataset_name
        for filter in self.__filters:
            curr_dataset_dir, num_classes = filter.process_dataset(curr_dataset_dir, curr_dataset_name)
            curr_dataset_name = os.path.basename(curr_dataset_dir)

        return curr_dataset_dir, num_classes


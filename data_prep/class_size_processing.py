import glob
import tqdm
import os
from data_prep.util import transfer_datapoints

class ClassSizeProcessing(object):
    def __init__(self, min_size, output_dataset_dir):
        self.__min_size = min_size
        self.__dest_root_dir = output_dataset_dir

    def process_dataset(self, raw_dataset_dir, dataset_name):
        num_classes = 0
        class_dirs = glob.glob(os.path.join(raw_dataset_dir, '*'))

        dest_ds_dir = os.path.join(self.__dest_root_dir, dataset_name + ' min_size=' + str(self.__min_size))

        if os.path.exists(dest_ds_dir):
            return dest_ds_dir, glob.glob(os.path.join(dest_ds_dir, '*'))
        for dir in tqdm.tqdm(class_dirs, desc='select classes by size'):
            datapoints = glob.glob(os.path.join(dir, '*'))
            if len(datapoints) >= self.__min_size:
                num_classes += 1
                cl = os.path.basename(dir)
                dest_cl_dir = os.path.join(dest_ds_dir, cl)
                transfer_datapoints(dest_cl_dir, dir, datapoints)

        return dest_ds_dir, num_classes

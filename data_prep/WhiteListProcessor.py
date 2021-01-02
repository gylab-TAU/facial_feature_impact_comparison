import glob
import tqdm
import os
from data_prep.util import transfer_datapoints


class WhiteListProcessor(object):
    def __init__(self, output_dataset_dir:str, white_list:list=[]):
        self.__output_dataset_dir = output_dataset_dir
        self.__white_list = white_list

    def process_dataset(self, raw_dataset_dir, dataset_name):
        num_classes = 0
        class_dirs = glob.glob(os.path.join(raw_dataset_dir, 'train', '*'))
        dest_ds_dir = os.path.join(self.__output_dataset_dir, dataset_name + ' white_list')

        if os.path.exists(dest_ds_dir):
            return dest_ds_dir, glob.glob(os.path.join(dest_ds_dir, '*"'))

        for dir in tqdm.tqdm(class_dirs, desc='white list'):
            cl = os.path.basename(dir)
            if cl in self.__white_list:
                train_datapoints = glob.glob(os.path.join(dir, '*'))
                num_classes += 1
                cl = os.path.basename(dir)
                dest_cl_dir = os.path.join(dest_ds_dir, cl)
                transfer_datapoints(dest_cl_dir, dir, train_datapoints)

        return dest_ds_dir, num_classes
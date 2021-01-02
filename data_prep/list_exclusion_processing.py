import glob
import tqdm
import os
from data_prep.util import transfer_datapoints


class ListExclusionProcessor(object):
    def __init__(self, output_dataset_dir:str, exclusion_list:list=[]):
        self.__output_dataset_dir = output_dataset_dir
        self.__exclusions = exclusion_list

    def process_dataset(self, raw_dataset_dir, dataset_name):
        num_classes = 0
        class_dirs = glob.glob(os.path.join(raw_dataset_dir, '*'))
        dest_ds_dir = os.path.join(self.__output_dataset_dir, dataset_name + ' exclude_from_list')

        if os.path.exists(dest_ds_dir):
            return dest_ds_dir, glob.glob(os.path.join(dest_ds_dir, '*"'))

        for dir in tqdm.tqdm(class_dirs, desc='excldue classes by list'):
            cl = str(int(os.path.basename(dir).replace('n', '')))
            if cl not in self.__exclusions:
                datapoints = glob.glob(os.path.join(dir, '*'))
                num_classes += 1
                cl = os.path.basename(dir)
                dest_cl_dir = os.path.join(dest_ds_dir, cl)
                transfer_datapoints(dest_cl_dir, dir, datapoints)
            elif cl in  self.__exclusions:
                print('found ', cl, ' in exclusions')

        return dest_ds_dir, num_classes
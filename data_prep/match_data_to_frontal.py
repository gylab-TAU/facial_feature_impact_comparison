'''for frontal faces - according to a csv with id folder and number of images in train [0] and val[0] makes the same
 dataset from the src_path (of a non-frontal dataset) to a dest_path which will have the same number of image for each id
 issue: if dest source folder contains less images than csv defines'''


import os, random
import shutil
import pandas as pd
from _collections_abc import Set as _Set, Sequence as _Sequence
import numpy as np


class match_num_to_frontal:

    def __init__(self, src_path, dst_path, csv_file):
        self.src_path = src_path
        self.csv_file = csv_file
        self.dst_path = dst_path

    def create_data(self):
        filename = self.csv_file
        id_dictio = {}
        train_or_val = ['train', 'val']
        chosen = []

        data = pd.read_csv(filename)
        for key, value in data.iteritems():
            id_dictio[key] = value
        print(id_dictio)
        for key in id_dictio:
            #dir names
            src_dir = os.path.join(self.src_path, train_or_val[0]) # train
            # choose a random id
            id_list = os.listdir(src_dir)
            set_a = set(id_list)
            set_b = set(chosen)

            # Get new set with elements that are only in id_list but have not yet been chosen
            only_in_id_list = list(set_a.difference(set_b))

            id_folder = random.choice(only_in_id_list)
            # id_folder = str(random.sample(only_in_id_list, 1)[0])
            #add id to chosen list
            chosen.append(id_folder)

            #paths
            dest_dir_train = os.path.join(self.dst_path, train_or_val[0], id_folder) # train
            dest_dir_val = os.path.join(self.dst_path, train_or_val[1], id_folder) # val

            src_dir_train = os.path.join(self.src_path, train_or_val[0], id_folder) #train
            src_dir_val = os.path.join(self.src_path, train_or_val[1], id_folder) #val

            #number of images to copy
            num_to_copy_train = id_dictio[key][0]
            num_to_copy_val = id_dictio[key][1]

            # make dir of id
            if not os.path.exists(dest_dir_train):
                os.makedirs(dest_dir_train)
            if not os.path.exists(dest_dir_val):
                os.makedirs(dest_dir_val)

            #choose num_to_copy random images
            filename_list_test = np.random.choice(os.listdir(src_dir_train), num_to_copy_train, replace=False)
            #val folder can be empty or with less than #num_to_copy_val images, if so - take from test folder a list of images that were not taken for filename_list_test- TODO: make a val folder with at least 10 images per id
            if len(os.listdir(src_dir_val)) < num_to_copy_val:
                val_dir_empty = 1
                #change src of val folder to test folder
                src_dir_val = os.path.join(self.src_path, train_or_val[0], id_folder)  # test

            else:
                val_dir_empty = 0
            filename_list_val = np.random.choice( list(set(os.listdir(src_dir_train)) - set(filename_list_test)) if val_dir_empty else os.listdir(src_dir_val), num_to_copy_val, replace=False)
            # filename_list_val = np.random.choice(os.listdir(src_dir_val), num_to_copy_val)

            #copy #dict[key][0] from srv/test/id to dst/test/id
            for filename_train in filename_list_test:
                src = os.path.join(src_dir_train, filename_train)
                dst = os.path.join(dest_dir_train, filename_train)
                # copy from src to dest
                shutil.copy(src ,dst)

            #copy #dict[key][1] from srv/val/id to dst/val/id
            for filename_val in filename_list_val:
                #copy from src to dest
                shutil.copy(os.path.join(src_dir_val, filename_val), os.path.join(dest_dir_val, filename_val))

if __name__ == '__main__':
    args = None
    src_path = r"/home/administrator/datasets/images_faces/faces_only"
    dst_path = r"/home/administrator/datasets/10_img_per_id_diff_non_frontal"
    csv_file = r"/home/administrator/datasets/processed/csvs/val_train_same_img_10_per_id.csv"

    new_obj = match_num_to_frontal(src_path, dst_path, csv_file)
    new_obj.create_data()



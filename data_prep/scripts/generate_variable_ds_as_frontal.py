import os
import glob
import shutil
from pathlib import Path
import numpy as np


def generate_ds(src_path, dst_path, variable_path):
    #for each id folder in train in frontal faces - create same id folder with same nyumber of images from variable
    src_folder_train = Path(src_path+'/train/')
    dst_path_train = os.path.join(dst_path, 'train')
    dst_path_val = os.path.join(dst_path, 'val')
    variable_path_train = os.path.join(variable_path, 'train')
    variable_path_val = os.path.join(variable_path, 'val')

    # # make dst train and val folder
    # if not os.path.isdir(variable_path_train):
    #     os.mkdir(variable_path_val)
    # # if not os.path.isdir(dst_id_val):
    # #     os.mkdir(dst_id_val)

    for id in os.listdir(src_folder_train):

        src_id_train = os.path.join(src_folder_train, id)
        # src_id_val = os.path.join(src_folder_val, id)

        dst_id_train = os.path.join(dst_path_train, id)
        dst_id_val = os.path.join(dst_path_val, id)
        variable_id_train = os.path.join(variable_path_train, id)
        variable_id_val =  os.path.join(variable_path_val, id)

        #make dst id folder
        if not os.path.isdir(dst_id_train):
            os.makedirs(dst_id_train)
        # if not os.path.isdir(dst_id_val):
        #     os.makedirs(dst_id_val)

        images = os.listdir(Path(src_id_train))
        number_of_train_img = len(images)
        #generate a new id folder in dst dataset from variable faces with same number of images
        img_list = np.random.choice(os.listdir(variable_id_train), number_of_train_img, replace=None)



        for img in img_list:
            #make the img path from variable folder
            variable_id_train_img = os.path.join(variable_id_train, img)
            variable_id_val_img = os.path.join(variable_id_val, img)

            # copy all imgs from variable to id dest
            shutil.copy(variable_id_train_img, dst_id_train)  # train
            #shutil.copytree(src_dir_val, dest_dir_val, dirs_exist_ok=True)  # val


if __name__ == '__main__':
    #frontal folder
    src_path = r'/home/administrator/datasets/processed/test_for_frontal'
    dst_path =r'/home/administrator/datasets/processed/test_for_frontal_dest'
    variable_path =r'/home/administrator/datasets/images_faces/faces_only_300'
    generate_ds(src_path, dst_path, variable_path)

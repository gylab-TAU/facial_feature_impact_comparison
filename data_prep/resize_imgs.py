'''resize images in a folder to be equal, according to the largest image'''

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os
import shutil
import csv
import itertools



"""
analyse the dataset -create a csv  with quantity of images for each id (val & train) & resize all images to fit mtcnn
"""


import os
from PIL import Image


# path = r'C:\Users\mandy\Desktop\studies\masterSupp\project\faceAlignment\test_source'
# path = r'/home/administrator/datasets/processed/same_imgs_10_per_id'
path = r'/home/administrator/datasets/id_changed_from_faces_only/50_ids_300_img_per_id'


csv_path = r'/home/administrator/datasets/processed/csvs/id_changed_from_faces_only_50_ids_300_imgs.csv'
dest_path = r'/home/administrator/datasets/id_changed_from_faces_only/500_ids_300_img_per_id'


def main(args):
    id_dic = {}
    test = os.path.join(path, 'train')
    val = os.path.join(path, 'val')
    folders = [test, val]
    max_width = 0
    max_height = 0

    for folder in folders:
        print(folder,':\n')
        id_list = os.listdir(folder) # dir is your directory path
        for id in id_list:
            id_path = os.path.join(folder ,id)
            if os.listdir(id_path):

                images_list = os.listdir(id_path)  # dir is your directory path
                number_files = len(images_list)
                id_dic.setdefault(id, []).append(number_files)
                print(id, ': ', number_files)
                # (height, width) = yield_largest(folder, id_path)
                # print('largest: ',(height, width))
                # (height, width) = (2637, 2529)
                # resize_to_largest(folder, id, (height, width))

    print(id_dic)
    # print('max size is: {max_height}, {max_width} size')
    #save dict to a csv
    with open(csv_path,'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(id_dic.keys())
        writer.writerows(itertools.zip_longest(*id_dic.values()))


def yield_largest(folder, id_path):

    #remove my-directory-list.txt
    text_file = os.path.join(id_path, 'my-directory-list.txt')
    if os.path.exists(text_file):
        os.remove(text_file)
    #get largest
    largest_image = max(Image.open(os.path.join(id_path, f), 'r').size for f in os.listdir(id_path))
    return largest_image

def resize_to_largest( folder,id, size):
    source_path = os.path.join(folder, id)
    # source_path.save(dest_path)
    orig_img_list = os.listdir(source_path)
    for img in orig_img_list:
        img_path = Image.open(os.path.join(source_path, img))
        new_image = img_path.resize(size)
        dst_path = os.path.join(dest_path, os.path.basename(folder), id)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        new_image.save(os.path.join(dst_path, img))
        # imag.pytorch.transform.resize(size).save(source_path, dest_path)
        # transforms.Resize(size)

    # torchvision.transform.resize(size).save(dest_path)


def copy_unprocessed():
    source = [ '/home/administrator/ datasets / processed /frontal_resized/val/n000029/0047_01.jpg', '/home/administrator/ datasets / processed /frontal_resized/val/n000029/0161_01.jpg', '/home/administrator/ datasets / processed /frontal_resized/val/n000029/0209_01.jpg', '/home/administrator/ datasets / processed /frontal_resized/val/n000029/0215_01.jpg', '/home/administrator/ datasets / processed /frontal_resized/val/n000029/0365_01.jpg', '/home/administrator/ datasets / processed /frontal_resized/val/n000029/0381_01.jpg']
    dest_dir = r'/home/administrator/datasets/processed/test_for_unprocessed_mtcnn/val/n000029'
    for i in range(len(source)):
        new_src = source[i].replace(" ", "")
        # new_dst = dest[i].replace(" ", "")
        new_dst = new_src.replace("frontal_resized", "test_for_unprocessed_mtcnn")
        print('copy ', new_src, ' to ', new_dst)
        if not os.path.exists(dest_dir):
            print('making ', dest_dir)
            os.mkdir(dest_dir)
        shutil.copy(new_src, new_dst)


if __name__ == '__main__':
    args = None;
    main(args)
    # copy_unprocessed()


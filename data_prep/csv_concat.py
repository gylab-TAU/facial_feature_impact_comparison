import os
import glob
from pathlib import Path
import re

import pandas as pd
import shutil
# os.chdir("/home/administrator/experiments/mandy_experiments_configurations/gaus_2_csv_to_concat/2_ids/")
import pandas as pd


class CsvEdit:

    def __init__(self, csv_list, dest_csv, csv_dir_path):
        self.csv_list = csv_list
        self.dest_csv = dest_csv
        self.src_dir = Path(csv_dir_path)

    def add_img_num(self):
        for csv_file in self.csv_list:
            #split name of file
            images_num = csv_file.split("_")[-1].split('.')[0]
            print(images_num)
            df = pd.read_csv(csv_file)
            df.insert(0, 'Images', images_num)

            # df["images"] = ""
            dest_name = csv_file.split('.')[0]+'_img_added'
            dst_name = dest_name +'.csv'
            df.to_csv(dst_name, index=False)

    def concat(self):

        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        all_filenames = self.csv_list

        #combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
        #export to csv
        combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
        print(os.getcwd())

    def reorder(self):

        # df = pd.read_csv('/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/combined_csv.csv')
        df = pd.read_csv('/home/administrator/experiments/mandy_experiments_configurations/gaus_2_csv_to_concat/2_ids/combined_csv.csv')
        # df = pd.read_csv('/home/administrator/experiments/2_ids/2_ids_20_img_per_id_val_50/vgg16/results/comparisons_with_fc7_linear_img_added.csv')

        df_reorder = df[['Images','Unnamed: 0.1', 'input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'type']]

        df_reorder.to_csv(self.dest_csv, index=False)

    def copy_all_csvs(self):

        i=0
        for file in os.listdir(self.src_dir):
            path_to_folder_csv = str(self.src_dir) +'/'+ file + '/vgg16'+'/results'
            for f in os.listdir(path_to_folder_csv):
                if (re.match('comparison_with', f) or re.match('comparisons_with', f)):
                    name_of_csv = f
            path_to_csv =os.path.join( path_to_folder_csv, name_of_csv) #'/comparison_with_fc7_linear.csv'
            new_name = str(self.dest_csv) + '/comparison_with_fc7_linear'+str(i)+'.csv'
            shutil.copy(path_to_csv, self.dest_csv)  # copy the file to destination dir

            os.rename(os.path.join(self.dest_csv, name_of_csv), new_name)  # rename
            i+=1


if __name__ == '__main__':
    #create a list of the csvs:
    # csv_dir_path ='/home/administrator/experiments/mandy_experiments_configurations/gaus_3_csv_to_concat/5_ids'
    # csv_dir_path = Path('/home/administrator/experiments/random_run_results/')
    csv_dir_path = Path('/home/administrator/experiments/all_ids/')


    csv_list = os.listdir(csv_dir_path)
    # csv_list_to_concat = ['/home/administrator/experiments/2_ids/2_ids_20_img_per_id_val_50/vgg16/results/comparisons_with_fc7_linear.csv']
    csv_list_to_concat = []
    for i in csv_list:
        csv_list_to_concat.append( os.path.join(csv_dir_path, i))
    print(csv_list_to_concat)
    # csv_list = ["/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/test1.csv",
    #             '/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/test2.csv',
    #             '/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/test3.csv']
    dest_csv = "/home/administrator/experiments/random_run_results/"
    dest_csv = "/home/administrator/experiments/"
    # dest_csv = "/home/administrator/experiments/2_ids/2_ids_20_img_per_id_val_50/vgg16/results/comparisons_with_fc7_linear_img_added_ordered.csv"


    new_obj = CsvEdit(csv_list_to_concat, dest_csv, csv_dir_path)
    new_obj.add_img_num()
    new_obj.copy_all_csvs()
    # new_obj.concat()
    # new_obj.reorder()
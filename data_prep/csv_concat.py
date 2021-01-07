import os
import glob
import pandas as pd
os.chdir("/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/")
import pandas as pd


class CsvEdit:

    def __init__(self, csv_list, dest_csv):
        self.csv_list = csv_list
        self.dest_csv = dest_csv

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

    def reorder(self):

        df = pd.read_csv('/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/combined_csv.csv')
        # df_reorder = df[['A', 'B', 'C', 'D', 'E']]  # rearrange column here
        df_reorder = df[['Images','Unnamed: 0', 'input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'type']]

        df_reorder.to_csv(self.dest_csv, index=False)

if __name__ == '__main__':
    #create a list of the csvs:
    csv_dir_path ='/home/administrator/experiments/mandy_experiments_configurations/csvs_to_concat/1000_ids'
    csv_list = os.listdir(csv_dir_path)
    # csv_list_to_concat = ['/home/administrator/experiments/mandy_experiments_configurations/csvs_to_concat/10_ids/comparisons_with_fc7_linear_10_ids_50.csv']
    csv_list_to_concat = []
    for i in csv_list:
        csv_list_to_concat.append( os.path.join(csv_dir_path, i))
    print(csv_list_to_concat)
    # csv_list = ["/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/test1.csv",
    #             '/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/test2.csv',
    #             '/home/administrator/experiments/50_ids/50_ids_100_img_per_id_val_50/vgg16/results/test3.csv']
    dest_csv = "/home/administrator/experiments/mandy_experiments_configurations/csvs_to_concat/1000_ids/1000_ids_all.csv"

    new_obj = CsvEdit(csv_list_to_concat, dest_csv)
    # new_obj.add_img_num()
    new_obj.concat()
    new_obj.reorder()
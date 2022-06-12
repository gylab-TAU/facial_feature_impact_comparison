"""for all the 30 runs on each condition (id num * images per id), we want
to take only the fc7 (placed in /home/hdd_storage/MR/results/asians/2_ids/small_ids_no_asians_2_ids_2_5_8/vgg16/results/mandy_run/mat_dom)
and makean average of all 30 runs of the same condition. this will result
in 5*8 (num of id: 2,5,10,50,100 and num of images per id: 1,5,10,20,50,100,200,300) = 40
mat_dom (similarity matrix), each ready to be run in the notebook and then calc the AUC"""

import os
# import tensorflow as tf
import numpy as np
import pandas as pd
import csv


def run_and_calc(path, num_ids, num_pics, suffix, folder_prefix, results_path):
    #"run on all ids in path"
    for ids in num_ids:
        id_path = os.path.join(path,str(ids) +'_ids' )
        for image in num_pics:
            folder_suffix = str(ids) +'_ids_' + str(ids) +'_'+ str(image) 

    #      "run on all indexes in id and image"
            for index in range(run_num):
                folder_last = folder_suffix + '_' + str(index)
                folder_name = folder_prefix+ folder_last 
                full_path_to_folder = os.path.join(id_path, folder_name)
                full_path_fc7 =full_path_to_folder+ suffix

                if index ==0 :
                    # data = np.recfromcsv(full_path_fc7, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
                    # print(data)


                    #reset addition - first df
                    with open(full_path_fc7,'r') as dest_f:
                        data_iter = csv.reader(dest_f,
                                            delimiter = ',',
                                            quotechar = "'")
                        data = [data for data in data_iter]
                    data_array = np.asarray(data).size
                    # print('data_array: \n',data_array)
                    print(data_array)


                    # # from numpy import genfromtxt
                    # my_data = np.genfromtxt(full_path_fc7, delimiter=',', dtype = None)
                    # print('my_data: \n',my_data)


#                     origin_df = pd.read_csv(full_path_fc7, header=0)
#                     print('origin_df: ',origin_df)
#                     print(origin_df[['Unnamed: 0']].to_string(index=False)) 
#                     origin_df = origin_df.set_index("Unnamed: 0")
#                     print('origin_df after set index: ',origin_df)

#                     #tamar_08.jpg.1
#                     # print('origin_df last col: ',origin_df.columns[153])
#                     #extract first column:
#                     first_col = origin_df.iloc[:,0]
#                     # extract first row:
#                     first_row = origin_df.columns#.iloc[0]
#                     # first_row = origin_df.index.tolist()
#                     print(type(first_row))
#                     # print('first_row: ', first_row.tolist())
#                 else: # addition for all 29 df
#                     #read csv as df
#                     df_to_add = pd.read_csv(full_path_fc7)
#                     #addition
#                     origin_df = origin_df.iloc[0:,1:].add(df_to_add.iloc[0:,1:])
#             #average     
#             origin_df = origin_df.div(run_num)
#             #add first column back
#             origin_df.insert(0, 'Name', first_col)
#             origin_df = origin_df.set_index( 'Name')

# #           "save average fc7 for id and image, with id and image in name"
#             result_path = os.path.join(results_path, folder_suffix+'.csv')
#             print(f'saved to {result_path}')
#             print(f'shape is : {origin_df.shape}')
#             # origin_df.to_csv(result_path)



if __name__ == "__main__":

    # params:
    path = "/home/hdd_storage/MR/results/asians"
    suffix = "/vgg16/results/mandy_run/mat_dom/mat_dom_fc7.csv"
    folder_prefix = "small_ids_no_asians_"
    results_path = "/home/mandy/project/facial_feature_impact_comparison/configs/ps_sr/results"
    run_num = 3
    num_ids = [2,5]
    num_pics = [1,5]
    #run the func
    run_and_calc(path, num_ids, num_pics, suffix, folder_prefix, results_path)

import os
import glob
import random
import pandas


#def create_main_dataset(root_dir, dest_dir, excluded_ids_num, min_img_num ):
#params:
orig_root_dir = '/home/administrator/datasets/vggface2_mtcnn/'
dest_root_dir = '/home/administrator/experiments/familiarity/dataset'
famous_ids_file = '/home/administrator/experiments/familiarity/famous_ids_libi.csv'
min_img_num = 320
random_seed = 0
excluded_ids_num = 300

dest_famous_dir = os.path.join(dest_root_dir, 'famous_fixed')
dest_pretraining_dataset_dir = os.path.join(dest_root_dir, 'pretraining_dataset_fixed')
dest_finetuning_dataset_dir = os.path.join(dest_root_dir, 'finetuning_dataset_fixed')

os.makedirs(dest_famous_dir, exist_ok=True)
os.makedirs(dest_pretraining_dataset_dir, exist_ok=True)
os.makedirs(dest_finetuning_dataset_dir, exist_ok=True)

random.seed(random_seed)

ids_df = pandas.read_csv(famous_ids_file)


low_img_num_list = []
high_img_num_list = []
famous_ids_list = []

entries = os.scandir(orig_root_dir)

for entry in entries:
    curr_orig_dir = entry.path

    if entry.name in ids_df.values:
        famous_ids_list.append(entry)
        curr_dest_dir = os.path.join(dest_famous_dir, entry.name)
        if not os.path.exists(curr_dest_dir):
            os.symlink(curr_orig_dir, curr_dest_dir, target_is_directory=True)
        continue

    if os.path.isdir(curr_orig_dir):
        images = glob.glob(os.path.join(curr_orig_dir, "*.jpg"))
        img_num = len(images)
        if img_num>=min_img_num:
            high_img_num_list.append(entry)
        else:
            low_img_num_list.append(entry)
            curr_dest_dir = os.path.join(dest_pretraining_dataset_dir, entry.name)
            if not os.path.exists(curr_dest_dir):
                os.symlink(curr_orig_dir, curr_dest_dir, target_is_directory=True)

excluded_ids_list = random.sample(high_img_num_list, excluded_ids_num)


for entry in high_img_num_list:
    if entry in excluded_ids_list:
        print("here")
        curr_dest_dir = os.path.join(dest_finetuning_dataset_dir, entry.name)
        if not os.path.exists(curr_dest_dir):
            os.symlink(entry.path, curr_dest_dir, target_is_directory=True)
    else:
        curr_dest_dir = os.path.join(dest_pretraining_dataset_dir, entry.name)
        if not os.path.exists(curr_dest_dir):
            os.symlink(entry.path, curr_dest_dir, target_is_directory=True)


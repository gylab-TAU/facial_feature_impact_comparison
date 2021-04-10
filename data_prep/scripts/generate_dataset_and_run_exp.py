'''generate a dataset according to parameters and run the experiment'''

from data_prep.change_id_num import ChangeId
from data_prep.id_num_change import IdNum

class RunAll():

    def __init__(self, num_of_ids, num_of_images, src_folder, dataset_folder, exp_folder, num_of_runs ):
        self.num_of_ids = int(num_of_ids)
        self.num_of_images = num_of_images
        self.src_folder = src_folder
        self.dataset_folder = dataset_folder
        self.exp_folder = exp_folder
        self.num_of_runs = int(num_of_runs)


    def make_dataset(self):
        # make number of id with 300 images per id
        new_obj1 = change_id(self.src_folder, self.num_of_ids, self.dataset_folder)
        new_obj1.make_id_folder()

        # makes a new folder with ids, each with random #num_img images - could be same images or different one
        src_path = r"/home/administrator/datasets/processed/500_ids_5_train_50_val"
        dst_path = r"/home/administrator/datasets/processed/500_ids_1_train_50_val"
        same = 0

        new_obj2 = id_num(self.dataset_folder, self.num_of_images, dst_path, same)
        new_obj2.multiply()


    # run the experiment with made dataset
    def run_exp(self):
        print('run exp on dataset')



    def main(self):
        #number of ids in train and val
        num_of_ids = 200
        #number of images per id
        num_of_images = [300]
        #the src folder from which the ids are taken
        src_folder = "/home/administrator/datasets/images_faces/faces_only_300"
        #tha dataset folder to be createed
        dataset_folder = "/home/administrator/datasets/processed/500_ids_300_train_50_val"
        #the experiment folder to be created
        exp_name = '500_ids_50_train'
        #number of times the dataset will be created with those parameters and number of times to run the experimernt (each time running once on each dataset)
        num_of_runs = 1


        for i in num_of_runs:
            #creating a new objuxt for each run
            new_run = RunAll(num_of_ids, num_of_images, src_folder, dataset_folder+'_'+i, exp_name+'_'+i)
            #make a dataset
            new_run.make_dataset(self)
            #run on dataset the experiment
            new_run.run_exp(self)

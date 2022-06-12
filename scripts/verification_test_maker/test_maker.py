"""create a random test for inversion/other-race effect tst according to the unlearned images from vggface2"""

import random
import os
import sys
import shutil
sys.path.append('../')
import facial_feature_impact_comparison.scripts.num2name.main as recognize
import facial_feature_impact_comparison.scripts.create_inverted as inversion

def make_test(large_dataset_path, training_set_list, name_list):
    chosen_list = []
    chosen_name_list = []
    while len(chosen_list) < 10:
        #randomly select an id from large dataset
        all_ids_list = os.listdir(large_dataset_path)
        # print(all_ids_list)
        id = random.choice(all_ids_list)
        # img_path = random.choice(os.listdir(os.join(large_dataset_path, id)))
        # print(img_path)
        id_path = os.path.join(large_dataset_path, id)
        if id not in chosen_list:
            print('1')
            #check the identity of this id (name) using model - only more than 0.9 assurance
            score, img_path, name = recognize.recognize_process(id_path)
            print('2')
            print(score, img_path, name)
            #check if this name is in the training set (mapping_list)
            with open(name_list, 'r') as names_list:
                content = names_list.read()
                print('content: ',content)

                if name[1:] not in content:# (training set)
                    # if not in training set - randomly select 2 images
                    #images for test
                    image1 = img_path
                    image2=image1
                    while image2 == image1:
                        image2 = os.path.join(id_path,random.choice(os.listdir(id_path) ))
                    print(f'image1: {image1}, image2: {image2}')
                    #copy images to "vgg_test_pairs"
                    img1_with_id = id+ '_' + os.path.basename(image1)
                    img2_with_id = id+ '_' + os.path.basename(image2)

                    # shutil.copy(image1,f'/home/ssd_storage/datasets/MR/vggface_test/asian_images_dataset/{img1_with_id}')
                    # shutil.copy(image2,f'/home/ssd_storage/datasets/MR/vggface_test/asian_images_dataset/{img2_with_id}')
                    # with open('/home/ssd_storage/datasets/MR/vggface_test/asian_images_pairs_list/asian_upright_test_pairs.txt', 'a') as upright_test_pairs:
                    #upright same
                        # upright_test_pairs.write( img1_with_id + ' ' +img2_with_id + ' 1\n')               
                    #inverted same
                    # with open('/home/ssd_storage/datasets/MR/vggface_test/asian_images_pairs_list/asian_inverted_test_pairs_new.txt', 'a') as inverted_test_pairs:

                    #     inverted_test_pairs.write( img1_with_id + ' '+ img2_with_id + ' 1\n')

                    #same
                    
                    #and different



                    
                    #add to chosen_list
                    chosen_list.append(id_path)
                    chosen_name_list.append(name)
                    print(f'len: {len(chosen_list)}')
                else:
                    print(f"it is already in training: {id}")
                    #if in training set - randomly choose another identity
                    continue
                print(id)
    print(f'chosen: {chosen_list}')
    print(f'chosen: {chosen_name_list}')
    print(f'same pairs are in: /home/ssd_storage/datasets/MR/vggface_test/asian_images_pairs_list/asian_test_pairs.txt')
    #invert all dataset
    # inversion.invert_dataset('/home/ssd_storage/datasets/MR/vggface_test/images_dataset', '/home/ssd_storage/datasets/MR/vggface_test/images_dataset')



def asian_folder_maker(large_dataset_path, training_set_list, name_list):
    chosen_list = []
    chosen_name_list = []
    count = 0

    all_ids_list = os.listdir(large_dataset_path)
    #run on all ids of 1000 ids:
    for id in all_ids_list:

        id_path = os.path.join(large_dataset_path, id)
        #check the identity of this id (name) using model - only more than 0.9 assurance #check assurance!
        score, img_path, name = recognize.recognize_process(id_path)
        print('2')
        print(score, img_path, name)
        print(f'is this the name wanted?: {name[3:-1]}')
        #check if this name is in the training set (mapping_list)
        with open(name_list, 'r') as names_list:
            content = names_list.read()
            # print('content: ',content)
            #if this is an asian (name in asian_list)
            if name[3:-1] in content:
                count +=1
                # # if not in training set - randomly select 2 images
                # #images for test
                # image1 = img_path
                # image2=image1
                # while image2 == image1:
                #     image2 = os.path.join(id_path,random.choice(os.listdir(id_path) ))
                # print(f'image1: {image1}, image2: {image2}')
                # #copy images to "vgg_test_pairs"
                # img1_with_id = id+ '_' + os.path.basename(image1)
                # img2_with_id = id+ '_' + os.path.basename(image2)

            #remove this id from no_asian_folder
                train_path = '/home/ssd_storage/datasets/processed/1000_ids_num_changed/1000_ids_300_train_50_val_no_asians/train'
                val_path  = '/home/ssd_storage/datasets/processed/1000_ids_num_changed/1000_ids_300_train_50_val_no_asians/val'
                id_train_path  = os.path.join(train_path, id)
                id_val_path  = os.path.join(val_path, id)
                print(f'id_train_path: {id_train_path}')
                print(f'id_val_path: {id_val_path}')
                shutil.rmtree(id_train_path)
                shutil.rmtree(id_val_path)

                # with open('/home/ssd_storage/datasets/MR/vggface_test/asian_images_pairs_list/asian_upright_test_pairs.txt', 'a') as upright_test_pairs:
                # #upright same
                #     upright_test_pairs.write( img1_with_id + ' ' +img2_with_id + ' 1\n')               
                #inverted same
                # with open('/home/ssd_storage/datasets/MR/vggface_test/asian_images_pairs_list/asian_inverted_test_pairs_new.txt', 'a') as inverted_test_pairs:

                #     inverted_test_pairs.write( img1_with_id + ' '+ img2_with_id + ' 1\n')

                #same
                
                #and different



                
                #add to chosen_list
                # chosen_list.append(id_path)
                # chosen_name_list.append(name)
                # print(f'len: {len(chosen_list)}')
            else:
                print(f"it is not an asian: {name}")
                #if in training set - randomly choose another identity
                continue
            print(id)
    print('count:' , count)
    # print(f'chosen: {chosen_list}')
    # print(f'chosen: {chosen_name_list}')
    # print(f'same pairs are in: /home/ssd_storage/datasets/MR/vggface_test/asian_images_pairs_list/asian_test_pairs.txt')
    #invert all dataset
    # inversion.invert_dataset('/home/ssd_storage/datasets/MR/vggface_test/images_dataset', '/home/ssd_storage/datasets/MR/vggface_test/images_dataset')



def names_extractor(training_set_list, name_list):
    """from a list with multiple arguments, extract only names"""
    with open(training_set_list, 'r') as full_list:
        with open(name_list, 'w') as name_list:
            for line in full_list:
                name = line.split()[1] #name in second place
                name_list.write(name + ' ')

def check_if_in_dataset():
    """checks if the identity chosen is in the training set"""
    print('h')

def make_couples():
    """make couples of images for same/different and frontal/inverted"""
    print('h')

def inverse():
    """inverse the image for inversion effect check"""
    print('h')

def is_asian():
    """checks if asian"""
    print('h')

if __name__ == "__main__":
    large_dataset_path = '/home/administrator/datasets/vggface2_mtcnn' #'/home/administrator/datasets/vgg_test_to_remove' #origin: '/home/administrator/datasets/vggface2_mtcnn'
    # large_dataset_path = '/home/ssd_storage/datasets/students/OM/datasets/2' #'/home/administrator/datasets/vgg_test_to_remove' #origin: '/home/administrator/datasets/vggface2_mtcnn'
    # large_dataset_path = '/home/ssd_storage/datasets/processed/1000_ids_num_changed/1000_ids_300_train_50_val/train'
    # training_set_list = '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/all_asians_id_mapping_list.txt'
    training_set_list = '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/asian_training.txt'

    # name_list = '/home/mandy/project/facial_feature_impact_comparison/scripts/verification_test_maker/text_files/all_ids_general_asian_name_list.txt'
    name_list = '/home/mandy/project/facial_feature_impact_comparison/scripts/verification_test_maker/text_files/name_list.txt'
    diff_pairs = ''# white_hv_diff_pairs_inverted.txt
    same_pairs = ''
    all_pairs = ''# white_hv_all_pairs.txt
    make_test(large_dataset_path, training_set_list, name_list)
    
    # names_extractor(training_set_list, name_list)
    # asian_folder_maker(large_dataset_path, training_set_list, name_list)
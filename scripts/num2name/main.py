"""for all 1000 ids in folder, map number of identity to name of identity. output a file with the mapping"""

import os
import random
import keras_vggface
import numpy as np
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


def run_all_ids(folder_path):
    """runs on all identities in 1000 ids folder"""
    ids_list = os.listdir(folder_path)
    # print(ids_list)
    for identity in os.listdir(folder_path):
        id_folder = os.path.join(folder_path , str(identity))
        img_path = os.path.join(id_folder, random.choice(os.listdir(id_folder)))
        print(img_path)
        with open('full_vggface2_images_to_check.txt', 'a') as images_to_recognize:
            images_to_recognize.write(img_path)
            images_to_recognize.write('\n')
        

def recognize(images_to_recognize):
    """for every image of id, recognize the name and make a new file with mapped num - name"""
    with open(images_to_recognize) as images_list:
        for img_path in images_list:
            print(img_path)
            recognizer(img_path=img_path)
            

def recognizer(img_path , model = VGGFace(model='resnet50')):
    #load model
    model = model #VGGFace(model='resnet50')
    # load the image
    img = image.load_img(img_path.strip(),target_size=(224, 224))

    # prepare the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)

    # perform prediction
    preds = model.predict(x)
    name =  utils.decode_predictions(preds)[0][0][0]
    score = utils.decode_predictions(preds)[0][0][1]
    listing = [name, score]
    if score < 0.98:
        print(f'**************lower:{score}****************' )

    #id_num extract
    id_num = id_num_extractor(img_path)
    # print('Predicted:', utils.decode_predictions(preds))
    with open('/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/vggface2_mapping_list.txt','a') as mapping_file:
        mapping_file.write(name + ' ' +str(score) + ' ' + id_num + '\n')


def id_num_extractor(img_path):
    """extract the number of id from the path, always the second before last in path"""
    path_argument = img_path.split('/')
    id_num = path_argument[-2]
    print(id_num)
    return id_num

def recognize_process(id_folder, model = VGGFace(model='resnet50') ):
    score = 0
    while score < 0.10:
        img_path = os.path.join(id_folder, random.choice(os.listdir(id_folder)))

        # prepare the image
        #recog
        img = image.load_img(img_path.strip(),target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)

        # perform prediction
        preds = model.predict(x)
        name =  utils.decode_predictions(preds)[0][0][0]
        score = utils.decode_predictions(preds)[0][0][1]
        listing = [name, score]
    return (score, img_path, name)


def run_all(folder_path):
    """runs on all identities in 1000 ids folder"""
    ids_list = os.listdir(folder_path)
    model = VGGFace(model='resnet50')
    score = 0
    print('first')
    print('folder_path: ', folder_path)
    print('os.listdir(folder_path): ', os.listdir(folder_path))
    for identity in os.listdir(folder_path):
        print('id: ', identity)
        id_folder = os.path.join(folder_path , str(identity))

        while score < 0.010:
            print('score: ', score)
            img_path = os.path.join(id_folder, random.choice(os.listdir(id_folder)))

            # prepare the image
            #recog
            img = image.load_img(img_path.strip(),target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)

            # perform prediction
            preds = model.predict(x)
            name =  utils.decode_predictions(preds)[0][0][0]
            score = utils.decode_predictions(preds)[0][0][1]
            listing = [name, score]
            print('listing:' , listing)

        with open('/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/all_ids_asian_mapping_list.txt','a') as mapping_file:
            mapping_file.write(name + ' '  + ' ' +str(score) + ' ' + identity + '\n')
        with open('asian_images_to_check.txt', 'a') as images_to_recognize:
            images_to_recognize.write(img_path)
            images_to_recognize.write('\n')
        score = 0



def get_asians():
    """for every folder, check the number of asians and who they are"""

    #check if name in 1000 ids list
    countin = 0
    countout = 0
    with open('/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/asian_mapping_list.txt', 'r') as asian_list:
        lines = [line for line in asian_list]
        for line in lines:
            print(f'{line}')
            asian_name = line.split("'")[1]
            score = line.split(" ")[-2]
            identity = line.split(" ")[-1] 
            print('asian_name: ', asian_name)
            with open( '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/mapping_list.txt', 'r') as all_ids:
                content = all_ids.read()
                if asian_name in content:
                    print(f'{asian_name} is in content, line: {line} ')
                    with open('/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/asian_training.txt','a' ) as asian_training:
                        asian_training.write(asian_name + ' ' + str(score) + ' ' + identity)
                    countin +=1
                else:
                    print(f'{asian_name} not in content')
                    with open('/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/asian_test.txt', 'a') as asian_test:
                        asian_test.write(asian_name + ' '  + str(score) + ' ' + identity )
                    countout +=1
    print(f' there are {countin} in the list and {countout} not in content')
            

def get_ids_not_in_folder(few_ids_file, all_ids_file, not_included_file,ids_folder_path):
    """compare two names files, and report what names from second file are not in the
    first one (in order to add ids to the 960 no asian folder), write in a txt file all
    ids that have more than 300 images in their folder"""
    #open three files
    with open(not_included_file, 'a') as not_included:
        with open(few_ids_file, 'r') as few_ids:
            contents_few_ids = few_ids.read()
            # print(contents_few_ids)
            with open(all_ids_file, 'r') as all_ids:         
            # run on all ids of larger file
                lines = [line for line in all_ids]
                for line in lines:
                    #extract name and id path:
                    name = line.split("'")[1]
                    # print(f' name: {name}')
                    id_num = line.split(" ")[3].split("\n")[0]
                    accuracy = line.split(" ")[2]

                    id_path = os.path.join(ids_folder_path,id_num )
                    # print(f' id_path: {id_path}')

                    #reset count:
                    count =0
                    #check if id in first file
                    if name not in contents_few_ids and (float(accuracy) > 0.95):
                        # print(f'name {name} not in content')
                        # if no: check if there are 300 images or more
                        for image in os.scandir(id_path):
                            if image.is_file():
                                count += 1           
                        # if yes: write to the not_included file (third file)
                        if count >400:
                            # print('count higher')
                            not_included.write(name + ' '+ id_path + '\n')
                #else continue
            #save txt file


def check_non_overlap(vggface2_file, lfw_file, not_included_file,ids_folder_path):
    """compare two names files, and report what names from second file are not in the
    first one (check if there is overlap), write in a txt file all
    ids that overlap"""
    #open three files
    with open(not_included_file, 'a') as included:
        with open(vggface2_file, 'r') as vggface2_ids:
            contents_vggface2_ids = vggface2_ids.read()
            # print(contents_few_ids)
            with open(lfw_file, 'r') as lfw_ids:         
            # run on all ids of larger file
                lines = [line for line in lfw_ids]
                for line in lines:
                    #extract name and id path:
                    # name = line.split("'")[1]
                    # print(f' name: {name}')
                    name = line.split("_")[0].split("/")[0].split("\n")[0]
                    full_name = name +"_"+ line.split("_")[1].split("/")[0].split("\n")[0]
                    # accuracy = line.split(" ")[2]

                    # id_path = os.path.join(ids_folder_path,id_num )
                    # print(f' id_path: {id_path}')

                    #reset count:
                    count =0
                    #check if id in first file
                    if name in contents_vggface2_ids:
                        # print(f'name {name} in content')
                        if full_name in contents_vggface2_ids:
                            print(f'full_name {full_name} in content')
                        # if no: check if there are 300 images or more
                        # for image in os.scandir(id_path):
                            # if image.is_file():
                                # count += 1           
                        # if yes: write to the not_included file (third file)
                        # if count >400:
                            # print('count higher')
                            included.write(full_name + '\n')
                #else continue
            #save txt file

def make_lfw_test_no_overlap_ids(origin_lfw_file, new_lfw_file, overlapping_ids ):
    """based on the lfw test, use the file with all ids that overlap the lfw
    and vggface2 dataset, and remove the overlapping ids,
    so the test is now with no ids trained on"""
    count = 0
    with open(overlapping_ids, 'r') as overlapping:
        overlapping_ids_cont = overlapping.read()

    with open(origin_lfw_file, "r") as origin_f:
        lines = origin_f.readlines()
    with open(new_lfw_file, "w") as new_f:
        for line in lines:
            name = line.split("/")[0].strip("\n")
            # print("name: ", name)
            if name not in overlapping_ids_cont:
                print("this: ", name)
                new_f.write(line)
            else:
                print(f"line {line} in overlapping")
                count+=1
    print(f'all names: {count}')


if __name__ == "__main__":
    # ids_folder_path = '/home/ssd_storage/datasets/processed/1000_ids_num_changed/1000_ids_300_train_50_val/train'
    ids_folder_path = '/home/hdd_storage/vggface2/vggface2_mtcnn'
    # asian_ids_folder_path = '/home/ssd_storage/datasets/students/OM/datasets/2'
    # images_to_test_path  = '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/images_to_check.txt'
    # images_to_test_path = '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/full_vggface2_images_to_check.txt'
    few_ids_file = '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/mapping_list.txt'
    all_ids_file = '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/vggface2_mapping_list.txt'
    not_included_file = '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/overlapping_vggf2_lfw.txt'
    # run_all_ids(ids_folder_path)
    # recognize(images_to_test_path)
    # run_all(asian_ids_folder_path)
    # get_asians()
    # get_ids_not_in_folder(few_ids_file, all_ids_file, not_included_file,ids_folder_path)
    lfw_file = '/home/mandy/project/facial_feature_impact_comparison/lfw_test_pairs.txt'
    # check_non_overlap(all_ids_file, lfw_file, not_included_file, ids_folder_path)
    lfw_file_no_overlap = '/home/mandy/project/facial_feature_impact_comparison/lfw_test_pairs_no_overlap_vggface2.txt'
    make_lfw_test_no_overlap_ids(lfw_file, lfw_file_no_overlap, not_included_file )
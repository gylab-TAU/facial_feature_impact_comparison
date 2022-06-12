""""from specific folder with identities, run on all and choose randomely X ids
for each id choose 2 images to add to test
for each 2 images - add to txt a couple of "same" pairs (same identity)
and a couple of diff pairs (diff identity)
anotate those are same (1) or different (0)"""
import os
import random





def test_maker(path, test_path, need_check):
    """makes a test with images pairs, randomly chosen from a directory of identities"""
    all_ids = []
    #run on large dataset
    [all_ids.append(x[0]) for i,x in enumerate(os.walk(path))]
    #for teh diff pairs:
    chosen_images = [] #tuples of pairs
    #choose X folders randomly
    chosen_ids = random.sample(all_ids, pairs_num)
    with open(test_path, 'a') as test:
        count = 0
        #for each folder (identity)
        for identity in chosen_ids:
            #if white test - need to cjeck not in train and not asian:
            if need_check:
                if not is_cosher(identity, '/home/mandy/project/facial_feature_impact_comparison/scripts/num2name/not_included.txt', test_path ):
                    continue #next id
            count += 1
            #choose 2 images randomly
            all_images = os.listdir(identity)
            chosen_imgs = random.sample(all_images, 2)
            print('chosen pairs: ', chosen_imgs)
            first_img_path = os.path.join(identity, chosen_imgs[0])
            second_img_path = os.path.join(identity, chosen_imgs[1])
            #add path to txt test - same
            test.write(first_img_path + ' ' + second_img_path + ' ' + '1' '\n')
            #add images to list of images as tuples
            chosen_images.append((first_img_path, second_img_path))
        if count > 0: #meaning there are cosher identities found
            #shuffle all pairs so they are different ids
            diff_pairs = shuffle_pairs(chosen_images)
            #add all images pairs shuffled to txt as diff:
            [test.write(pair[0] + ' ' + pair[1] + ' ' + '0' '\n') for pair in diff_pairs]
            print(f'test saved in: {test_path}, there are {pairs_num-count} more ids to add')


def shuffle_pairs(pairs):
    result = []
    for pair in pairs:
        # Choose a random pair from the list that is not the same as the current pair
        random_pair = random.choice([p for p in pairs if p != pair and p not in result])
        # Add the new pair to the result list
        result.append((pair[0], random_pair[1]))
    return result

def is_cosher(identity, not_included, white_test):
    """check if the id chosen is cosher:
    1. is not in the training (hence in the "not_included" path)
    2. is in the "not_included" txt but does not have * at the end
    3. is not asian
    """
    path_to_asian_dataset = '/home/ssd_storage/datasets/MR/vggface_test/asian_images_dataset/2'
    with open(white_test,'r') as white_test:
        contents_white_test = white_test.read()
        with open(not_included,'r') as not_in:
            contents_not_in = not_in.read()
            #check in txt file, but without *, and not already chosen in the test
            if identity in contents_not_in and str(identity+' *') not in contents_not_in and identity not in contents_white_test:
                print('in not included, without *!')
                #check not asian: check the id is not in the asian dataset
                path_to_id = os.path.join(path_to_asian_dataset, identity.split('/')[-1])
                print('path_to_id: ', path_to_id)
                if os.path.exists(path_to_id):
                    return False #(not cosher- asian)
                else:
                    return True 







if __name__ == "__main__":
    #params:
    path_to_asian_dataset = '/home/ssd_storage/datasets/MR/vggface_test/asian_images_dataset/2'
    path_to_asian_test = '/home/ssd_storage/datasets/MR/vggface_test/asian_images_pairs_list/asian_upright_test_large_num_all_pairs.txt'
    path_to_white_dataset = '/home/hdd_storage/vggface2/vggface2_mtcnn'
    path_to_white_test = '/home/ssd_storage/datasets/MR/vggface_test/images_pairs_list/white_upright_test_large_num_all_pairs.txt'
    need_check = False #False for making Asians test, True if making white dataset (need to check not asian and not in training)
    #don't forget to make diff and same txt pairs from the *all pairs* test
    pairs_num = 211 #numbers of pairs to be in the test
    test_maker(path_to_asian_dataset, path_to_asian_test,need_check)
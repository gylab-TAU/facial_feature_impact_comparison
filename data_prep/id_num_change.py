''' makes a new folder with ids, each with random #num_img images - could be same images or different one'''
import os, random
import shutil
import argparse



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_path", type=str, default=None, help='A path to the src path we want to copy from')
    parser.add_argument("--dst_path", type=str, default=None, help='A path to the dest path we want to copy to')

    args = parser.parse_args()
    return args



class IdNum:




    def __init__(self, src_path, num_imgs, dst_path, same):
        self.src_path = src_path
        self.num_imgs = num_imgs
        self.dst_path = dst_path
        self.same = same


    def multiply(self):
        # parser
        train = os.path.join(self.src_path, 'train')
        val = os.path.join(self.src_path, 'val')
        folders = [train, val]
        num_to_copy = 1 if self.same else self.num_imgs

        #open the folder from which we will copy ids
        for folder in folders:
            id_list = os.listdir(folder)  # dir is the directory path
            for id in id_list:
                img_path = os.path.join(self.src_path, folder, id)
                train_or_val = os.path.basename(folder)
                dst = os.path.join(self.dst_path, train_or_val, id)

                #make dest
                if not os.path.isdir(dst):
                    os.makedirs(dst)


                if not self.same:
                    #choose randomly num_imgs from src folder to be copied to dest folder
                    if num_to_copy > len(os.listdir(img_path)):
                        num_to_copy = len(os.listdir(img_path))
                    filename_list = random.sample(os.listdir(img_path), num_to_copy)
                    for filename in filename_list:
                        filename, file_extension = os.path.splitext(filename)
                        dest_image = os.path.join(dst, filename + file_extension)
                        shutil.copy(os.path.join(img_path, filename + file_extension), dest_image)
                else:
                    #choose randomely one image from src folder to be copied num_imgs times to dest folder
                    filename, file_extension = os.path.splitext(random.choice(os.listdir(img_path)))
                    # copy the one image num_imgs time
                    for i in range(0, num_imgs):
                        dest_image = os.path.join(dst, filename + str(i)+file_extension)
                        shutil.copy(os.path.join(img_path, filename+file_extension), dest_image)



        print('copied',self.num_imgs ,'random images from each id to', self.dst_path)


if __name__ == '__main__':
    args = get_args()
    if args.src_path is not None:
        src_path =args.src_path
        dst_path = args.dst_path
    else:
        print('please enter src and dest paths')

    # args = None
    # src_path = r"/home/administrator/datasets/processed/100_ids_num_changed/100_ids_50_train_50_val"
    # dst_path = r"/home/administrator/datasets/processed/2_ids_num_changed_1/2_ids_50_train_50_val"
    same = 0
    num_imgs = 50

    new_obj = IdNum(src_path, num_imgs, dst_path, same)
    new_obj.multiply()

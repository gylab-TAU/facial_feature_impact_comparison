
import os, random
import shutil


class id_num:

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
                    filename_list = random.sample(os.listdir(img_path), num_to_copy)
                    for filename in filename_list:
                        filename, file_extension = os.path.splitext(random.choice(os.listdir(img_path)))
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
    args = None;
    src_path = r"/home/administrator/datasets/processed/mtcnn_frontal_resized_per_folder"
    dst_path = r"/home/administrator/datasets/same_imgs"
    same = 1
    num_imgs =10

    new_obj = id_num(src_path, num_imgs, dst_path, same)
    new_obj.multiply()

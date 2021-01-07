import os

from PIL import Image, ImageFilter
import ntpath
from pathlib import Path



class Blur():

    def __init__(self, src_image_path, src_path, dst_path, dst_image_path, blur_tech, blur_level):
        self.src_image_path = src_image_path
        self.src_path = src_path
        self.dst_image_path = dst_image_path
        self.dst_path = dst_path
        self.blur_tech = blur_tech
        self.blur_level = blur_level


    def blur_image(self):

        # Open existing image
        # OriImage = Image.open(self.src_image_path)
        # OriImage = Image.open(self.src_path)


        for image in Path(self.src_path).iterdir():

            if os.path.isdir(image):
                continue

            # Open existing image
            OriImage = Image.open(image)

            if self.blur_tech == 'Gaus':
                # Applying GaussianBlur filter
                blurImage = OriImage.filter(ImageFilter.GaussianBlur(self.blur_level))


            if self.blur_tech == 'Box':
                # Applying BoxBlur filter
                blurImage = OriImage.filter(ImageFilter.BoxBlur(self.blur_level))


            if self.blur_tech == 'Simple':
                # Applying simple blur filter
                blurImage = OriImage.filter(ImageFilter.BLUR)


            # Save Blur Image
            # blurImage.save(self.dst_image_path)
            dest_path = os.path.join(self.dst_path,os.path.basename(image))
            print(dest_path)
            blurImage.save(dest_path)
            print('blurred image saved to : ', dest_path)


if __name__ == '__main__':
    src_image_path = r'/home/administrator/datasets/high_low_ps_images/joined/CM46_2.png'
#####head views#######
    #half left
    # src_path = r'/home/administrator/datasets/faces_in_views/half-left/cropped'
    # src_path = r'/home/administrator/datasets/faces_in_views/half-left/cropped/bb'
    # src_path = r'/home/administrator/datasets/faces_in_views/half-left/cropped/bb/aligned'
    # src_path = r'/home/administrator/datasets/faces_in_views/half-left/cropped/bb/mtcnn_160/aligned'
    #frontal
    # src_path = r'/home/administrator/datasets/faces_in_views/frontal/'
    # src_path = r'/home/administrator/datasets/faces_in_views/frontal/aligned'
    # src_path = r'/home/administrator/datasets/faces_in_views/frontal/aligned/for-hl'
    # src_path = r'/home/administrator/datasets/faces_in_views/frontal/mtcnn_160/aligned'
    #quarter left
    # src_path = r'/home/administrator/datasets/faces_in_views/quarter-left/cropped'
    # src_path = r'/home/administrator/datasets/faces_in_views/quarter-left/cropped/bb'
    # src_path = r'/home/administrator/datasets/faces_in_views/quarter-left/cropped/bb/aligned'
    #ref
    # src_path = r'/home/administrator/datasets/faces_in_views/ref'
    # src_path = r'/home/administrator/datasets/faces_in_views/ref/aligned'
    src_path = r'/home/administrator/datasets/faces_in_views/ref/mtcnn_160/aligned'

####critical features#####
    # src_path = r'/home/administrator/datasets/high_low_ps_images/joined'

    dst_image_path = r'/home/administrator/datasets/processed/blurred/blurred_Gaus_3.jpg'
#####head views#######
    #half left
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/half-left/cropped'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/half-left/cropped/bb'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/half-left/cropped/bb/aligned'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/half-left/cropped/bb/mtcnn_160/aligned'
    #frontal
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/frontal'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/frontal/aligned'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/frontal/aligned/for-hl'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/frontal/mtcnn_160/aligned'
    #quarter left
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/quarter-left/cropped'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/quarter-left/cropped/bb'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/quarter-left/cropped/bb/aligned'
    #ref
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/ref'
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/ref/aligned'
    dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/faces_in_views/ref/mtcnn_160/aligned'

#####critical features######
    # dst_path = r'/home/administrator/datasets/processed/blurred/blurred_gaus_3/high_ps_low_ps/'

    blur_tech = 'Gaus' #/Simple/Box/Gaus
    blur_level = 3

    new_blur = Blur(src_image_path, src_path, dst_path, dst_image_path, blur_tech, blur_level)
    new_blur.blur_image()

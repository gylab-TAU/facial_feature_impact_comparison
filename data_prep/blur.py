from PIL import Image, ImageFilter


class Blur():

    def __init__(self, src_image_path, dst_image_path, blur_tech, blur_level):
        self.src_image_path = src_image_path
        self.dst_image_path = dst_image_path
        self.blur_tech = blur_tech
        self.blur_level = blur_level


    def blur_image(self):

        # Open existing image
        OriImage = Image.open(self.src_image_path)
        # OriImage.show()

        if self.blur_tech == 'Gaus':
            # Applying GaussianBlur filter
            blurImage = OriImage.filter(ImageFilter.GaussianBlur(self.blur_level))
            # gaussImage.show()

        if self.blur_tech == 'Box':
            # Applying BoxBlur filter
            blurImage = OriImage.filter(ImageFilter.BoxBlur(self.blur_level))
            # blurImage.show()

        if self.blur_tech == 'Simple':
            # Applying simple blur filter
            blurImage = OriImage.filter(ImageFilter.BLUR)
            # blurImage.show()

        # Save Blur Image
        blurImage.save(self.dst_image_path)


if __name__ == '__main__':
    src_image_path = r'/home/administrator/datasets/high_low_ps_images/joined/CM46_2.png'
    dst_image_path = r'/home/administrator/datasets/processed/blurred/blurred_Gaus_15.jpg'
    blur_tech = 'Gaus' #/Simple/Box/Gaus
    blur_level = 15

    new_blur = Blur(src_image_path, dst_image_path, blur_tech, blur_level)
    new_blur.blur_image()

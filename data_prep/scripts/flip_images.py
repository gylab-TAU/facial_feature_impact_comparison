import cv2
import os
import glob


def iterate_path(path):
    return glob.glob(os.path.join(path, '*'))


def flip_dir(input_images, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for im_path in input_images:
        print('im_path: ', im_path)
        image = cv2.imread(im_path)
        flipped_image = cv2.flip(image, 1)
        rel_im_name = os.path.relpath(im_path, input_dir)
        output_path = os.path.join(output_dir, rel_im_name)
        cv2.imwrite(output_path, flipped_image)
        cv2.imshow('Image', flipped_image)


if __name__ == '__main__':
    # input_dir = '/home/administrator/datasets/faces_in_views/half-left/cropped/bb/aligned/'
    input_dir = '/home/ssd_storage/datasets/MR/vggface_test/images_dataset/new_to_delete'
    output_dir = '/home/ssd_storage/datasets/MR/vggface_test/images_dataset/new_flipped_to_delete'
    # output_dir = '/home/administrator/datasets/faces_in_views/half-right/cropped/bb/aligned/'
    images = iterate_path(input_dir)
    flip_dir(images, input_dir, output_dir)
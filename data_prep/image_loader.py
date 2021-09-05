import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_prep.Datasets.activations_dataset import ActivationsDatasets
from PIL import Image
import torch
import const


class ImageLoader(object):
    def __init__(self, im_size, post_crop_size, dataset_mean, dataset_std, target_transform=None, crop_scale=None):
        normalize = transforms.Normalize(dataset_mean, dataset_std)
        self.target_transform = target_transform
        self.center_crop_tt = transforms.Compose([
                transforms.Resize(im_size),

                transforms.CenterCrop(post_crop_size),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=1.0),
                normalize,
            ])
        # if crop_scale is None:
        # crop_scale = (1.0, 1.0)
        # ratio = (1.0, 1.0)
        # , scale = crop_scale, ratio = ratio
        # RandomResizedCrop
        self.random_crop_tt = transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(256),  # For dlib -> mtcnn scaling
                #transforms.Resize([350, 350]), transforms.CenterCrop(256),
                transforms.RandomResizedCrop(post_crop_size),
                transforms.RandomCrop(post_crop_size),
                transforms.CenterCrop(post_crop_size),
                # transforms.RandomResizedCrop(post_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    # TODO: Add a load dir function
    # TODO: Add a load order for debugging purposes
    def load_dataset(self, dir_path, center_crop=True, with_path=False):
        """Loads a dataset based on a specific structure (dataset/classes/images)"""
        tt = self.random_crop_tt
        if center_crop:
            tt = self.center_crop_tt
        if with_path:
            return ActivationsDatasets(dir_path, tt, self.target_transform)
        return datasets.ImageFolder(dir_path, tt, target_transform=self.target_transform)

    def load_image(self, image_path, center_crop=True):
        if center_crop:
            tt = self.center_crop_tt
        else:
            tt = self.random_crop_tt

        im1 = Image.open(image_path)
        # fix bug from grayscale images
        # duplicate to make 3 channels
        if im1.mode != 'RGB':
            im1 = im1.convert('RGB')

        im1t = tt(im1)
        im1t = im1t.unsqueeze(0)

        if torch.cuda.is_available() and const.DEBUG is False:
            im1t = im1t.cuda()
        return im1t

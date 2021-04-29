import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import torch


class ImageLoader(object):
    def __init__(self, im_size, post_crop_size, dataset_mean, dataset_std, target_transform=None, crop_scale=None):
        normalize = transforms.Normalize(dataset_mean, dataset_std)
        self.target_transform = target_transform
        self.center_crop_tt = transforms.Compose([
                transforms.Resize([im_size, im_size]),
                transforms.CenterCrop(post_crop_size),
                transforms.ToTensor(),
                normalize,
            ])
        # if crop_scale is None:
        # crop_scale = (1.0, 1.0)
        # ratio = (1.0, 1.0)
        # , scale = crop_scale, ratio = ratio
        # RandomResizedCrop
        self.random_crop_tt = transforms.Compose([
                transforms.Resize([im_size, im_size]),
                transforms.RandomCrop(post_crop_size),
                transforms.CenterCrop(post_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    # TODO: Add a load dir function
    # TODO: Add a load order for debugging purposes
    def load_dataset(self, dir_path, center_crop=True):
        """Loads a dataset based on a specific structure (dataset/classes/images)"""
        if center_crop:
            return datasets.ImageFolder(dir_path, self.center_crop_tt, target_transform=self.target_transform)
        return datasets.ImageFolder(dir_path, self.random_crop_tt, target_transform=self.target_transform)

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

        if torch.cuda.is_available():
            im1t = im1t.cuda()
        return im1t

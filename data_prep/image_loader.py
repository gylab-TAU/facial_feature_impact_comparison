import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import torch


class ImageLoader(object):
    def __init__(self, im_size, dataset_mean, dataset_std):
        normalize = transforms.Normalize(dataset_mean, dataset_std)
        self.center_crop_tt = transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                normalize,
            ])
        self.random_crop_tt = transforms.Compose([
                transforms.RandomResizedCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    def load_dir(self, dir_path, center_crop=True):
        if center_crop:
            return datasets.ImageFolder(dir_path, self.center_crop_tt)
        return datasets.ImageFolder(dir_path, self.random_crop_tt)

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

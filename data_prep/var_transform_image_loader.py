import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data_prep.Datasets.activations_dataset import ActivationsDatasets
from data_prep.Datasets.triplet_dataset import TripletDataset

from PIL import Image
import torch
import const
from const import TRAIN_PHASE, TEST_PHASE


class ImageLoader(object):
    def __init__(self, source_transforms, target_transform=None):
        self.target_transform = target_transform
        self.test_transforms = source_transforms[TEST_PHASE]
        self.train_transforms = source_transforms[TRAIN_PHASE]

    # TODO: Add a load dir function
    # TODO: Add a load order for debugging purposes
    def load_dataset(self, dir_path, test=True, with_path=False, triplet=False):
        """Loads a dataset based on a specific structure (dataset/classes/images)"""
        tt = self.train_transforms
        if test:
            tt = self.test_transforms
        if triplet:
            return TripletDataset(dir_path, tt)
        if with_path:
            return ActivationsDatasets(dir_path, tt, self.target_transform)
        return datasets.ImageFolder(dir_path, tt, target_transform=self.target_transform)

    def load_image(self, image_path, test=True):
        if test:
            tt = self.test_transforms
        else:
            tt = self.train_transforms

        im1 = Image.open(image_path)
        # fix bug from grayscale images
        # duplicate to make 3 channels
        if im1.mode != 'RGB':
            im1 = im1.convert('RGB')

        im1t = tt(im1)
        im1t = im1t.unsqueeze(0)

        # if torch.cuda.is_available() and const.DEBUG is False:
        #     im1t = im1t.cuda()
        return im1t

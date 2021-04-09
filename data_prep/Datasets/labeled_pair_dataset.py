import torch.utils.data
import os
from torchvision.datasets.folder import default_loader
from data_prep.Datasets.utils import get_bad_indices


class LabeledComparisonDataset(torch.utils.data.Dataset):
    """
    dataset for comparing pairs of images with a pairs label (i.e. same=1, diff=0).
    self[idx] -> [image1], [label1], [image2], [label2], [pairs_label]
    Might raise FileNotExistsError
    """
    def __init__(self, dir_path: str, labeled_pairs_file_path: str, pairs_separator: str = ' ', transforms=None):
        """

        :param dir_path: path to the image dir
        :param labeled_pairs_file_path: path to the pairs file
        :param pairs_separator: the separator of images paths
        :param transforms: the transforms to use
        """
        labels_list = []
        pairs_list = []
        with open(labeled_pairs_file_path, 'r') as f:
            for line in f:
                labeled_pair = line.split(' ')
                labeled_pair[2] = int(labeled_pair[2].replace(os.linesep, ''))
                pairs_list.append(labeled_pair[:2])
                labels_list.append(labeled_pair[2])
        bad_indices = get_bad_indices(pairs_list)
        bad_indices.sort(reverse=True)
        for i in bad_indices:
            del pairs_list[i]
            del labels_list[i]

        self.__init__(dir_path, pairs_list, labels_list, transforms)

    def __init__(self, dir_path: str, pairs_list: list, labels_list: list, transforms=None):
        """

        :param dir_path: the image folder path
        :param pairs_list: the list of image pairs
        :param labels_list: the list of pairs labels
        :param transforms: the transforms to apply on the images
        """
        self.dir_path = dir_path
        self.transforms = transforms
        self.pairs_list = pairs_list
        self.labels_list = labels_list
        self.loader = default_loader

    def __getitem__(self, idx):
        label1 = os.path.basename(self.pairs_list[idx][0])
        label2 = os.path.basename(self.pairs_list[idx][1])
        try:
            im1 = self.__load_image(idx, self.pairs_list[idx][0])
            im2 = self.__load_image(idx, self.pairs_list[idx][1])
            return im1, label1, im2, label2, self.labels_list[idx]
        except:
            raise FileNotFoundError(f'error on {label1}, {label2}')

    def __load_image(self, idx, im_label):
        path = os.path.join(self.dir_path, )
        im = self.loader(path)
        if self.transforms is not None:
            im = self.transforms(im)
        return im

    def __len__(self):
        return len(self.pairs_list)

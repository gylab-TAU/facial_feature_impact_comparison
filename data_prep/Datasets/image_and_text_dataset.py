import torch
from torchvision.datasets import ImageFolder
import csv


class ImageAndTextDataset(ImageFolder):
    """
    A dataset for retrieving images with text vectors
    """

    def __init__(self, path, transforms, target_transforms, vector_csv_path):
        """
        __init__ is run once upon initializing the Dataset object
        :param vector_csv_path: Path to a csv file representing a list of contextual vectors
        where the index is the label
        label | context vector of label
        context_vectors: a dict based on the vector_csv_path:
            { label : vector of career } for each career
        """
        super().__init__(path, transforms, target_transform=target_transforms)
        self.context_vectors = dict()
        with open(vector_csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                label = row[0]
                self.context_vectors[int(label)] = torch.Tensor([float(i) for i in row[1:]])

    def __getitem__(self, idx):
        """
        Loads a sample from the dataset at the given :param: idx
        :return: the loaded sample (image, label, contextual vector)
        """
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, self.context_vectors[label]
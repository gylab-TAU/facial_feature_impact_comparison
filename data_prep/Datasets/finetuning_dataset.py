import torch.utils.data
import torchvision.datasets


class FinetuningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torchvision.datasets.ImageFolder, prior_dataset_length):
        self.inner = dataset
        self.prior_dataset_length

    def __getitem__(self, idx):
        image, cl = self.inner[idx]
        return image, cl+self.prior_dataset_length

    def __len__(self):
        return len(self.inner)

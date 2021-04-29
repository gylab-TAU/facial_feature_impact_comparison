import torch.utils.data
import glob
import os


class ComparisonDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, image_loader):
        self.dir_path = dir_path
        self.image_loader = image_loader
        self.classes = glob.glob(os.path.join(dir_path, '*'))
        self.images = []
        for cl in self.classes:
            self.images = self.images + glob.glob(os.path.join(cl, '*'))

    def __getitem__(self, idx):
        return self.image_loader.load_image(self.images[idx]), os.path.relpath(self.images[idx], self.dir_path)

    def __len__(self):
        return len(self.images)
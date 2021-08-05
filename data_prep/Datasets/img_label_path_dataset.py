from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision import datasets


class ImgLabelPathDataset(datasets.ImageFolder):
    """
    dataset retrieving images and their paths
    """

    def __init__(self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = datasets.folder.default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImgLabelPathDataset, self).__init__(root, transform,
                                          target_transform,
                                          loader,
                                          is_valid_file)
        """
        Args:
            dataset_dir_path = path to directory containing class dir
            img_transforms = transformations for the images
        """

    def __len__(self):
        return super(ImgLabelPathDataset, self).__len__()

    def __getitem__(self, idx):
        img_label = super(ImgLabelPathDataset, self).__getitem__(idx)
        return img_label[0], img_label[1], self.samples[idx][0]

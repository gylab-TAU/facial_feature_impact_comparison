from facenet_pytorch import MTCNN, training
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import glob
import tqdm

class MTCNNAlignProcessor(object):
    """
    Processor that crops aligned faces from face images (using trained MTCNN algo)
    """
    def __init__(self, image_size: int,
                 margin: int,
                 output_dataset_dir: str,
                 class_name_filter: str = '*',
                 batch_size: int = 16):
        """

        :param image_size: num of pixel to save the image (image_size x image_size)
        :param margin: margin from the bounding box to crop the face with
        :param output_dataset_dir: the root output dir to save the aligned dataset.
                    will save to ('output_dataset_dir/mtcnn_aligned/dataset_name')
        :param class_name_filter: the glob filter for the classes (default as *)
        :param batch_size: the number of images to activate MTCNN over (default 16)
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__mtcnn = MTCNN(image_size=image_size,
                             margin=margin,
                             device=device,
                             selection_method='center_weighted_size')
        self.__batch_size = batch_size
        self.__class_name_filter = class_name_filter
        self.__output_dataset_dir = output_dataset_dir

    def process_dataset(self, raw_dataset_dir, dataset_name):
        processed_dataset_output_path = os.path.join(self.__output_dataset_dir, 'mtcnn_aligned', dataset_name)

        orig_img_ds = datasets.ImageFolder(raw_dataset_dir, transform=None)

        orig_img_ds.samples = [
            (p, p)
            for p, _ in orig_img_ds.samples
        ]

        loader = DataLoader(
            orig_img_ds,
            num_workers=self.__workers,
            batch_size=self.__batch_size,
            collate_fn=training.collate_pil
        )

        for i, (x, b_paths) in tqdm.tqdm(enumerate(loader), desc='MTCNN align'):
            try:
                crops = [p.replace(raw_dataset_dir, processed_dataset_output_path) for p in b_paths]
                self.__mtcnn(x, save_path=crops)
                print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
            except TypeError:
                print('Bad paths: ', b_paths)

        num_classes_to_use = glob.glob(processed_dataset_output_path, self.__class_name_filter)

        return processed_dataset_output_path, num_classes_to_use


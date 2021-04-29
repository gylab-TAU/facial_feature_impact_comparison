from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os

if __name__ == '__main__':
    src_dir = r'/home/administrator/ datasets / processed /frontal_resized'
    dest_dir = r'/home/administrator/ datasets / processed /mtcnn_frontal_resized_per_folder'
    batch_size = 16
    workers = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    test = os.path.join(src_dir, 'train')
    val = os.path.join(src_dir, 'val')
    folders = [test, val]

    for folder in folders:
        print(folder, ':\n')
        id_list = os.listdir(folder)
        for id in id_list:
            id_path = os.path.join(folder, id)
            if os.listdir(id_path):
                mtcnn = MTCNN(
                    # image_size=112,
                    margin=0,
                    device=device,
                    selection_method='center_weighted_size'
                )

                orig_img_ds = datasets.ImageFolder(id_path, transform=None)


                orig_img_ds.samples = [
                    (p, p)
                    for p, _ in orig_img_ds.samples
                ]

                loader = DataLoader(
                    orig_img_ds,
                    num_workers=workers,
                    batch_size=batch_size,
                    collate_fn=training.collate_pil
                )

                crop_paths = []
                box_probs = []

                for i, (x, b_paths) in enumerate(loader):
                    try:
                        crops = [p.replace(src_dir, dest_dir) for p in b_paths]
                        mtcnn(x, save_path=crops)
                        crop_paths.extend(crops)
                        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
                    except TypeError:
                        print('Bad paths: ', b_paths)

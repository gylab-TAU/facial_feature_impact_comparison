from scipy.spatial.distance import squareform, pdist
import torch
from representation.acquisition.representation_save_hook import FileSystemHook
from representation.acquisition.representation_extraction import RepresentationExtractor

import pandas as pd
import os.path
from tqdm import tqdm
import torch.utils.data as data
from data_prep.Datasets.comparison_dataset import ComparisonDataset
import const


class EfficientRDM(object):
    def __init__(self, reps_cache_path, image_loader, get_layers_dict):
        self.__reps_cache_path = reps_cache_path
        self.__image_loader = image_loader
        self.__get_layers_dict = get_layers_dict

    def compare_pairs(self, model, dataset_dir, progress_label):
        dataset = ComparisonDataset(dataset_dir, self.__image_loader)
        layers_reps = {}
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
        re = RepresentationExtractor(model,
                                     self.__get_layers_dict(model),
                                     FileSystemHook(self.__get_layers_dict(model), self.__reps_cache_path))
        layers_reps = {}
        keys = []
        for l in self.__get_layers_dict(model).values():
            if l == 'model':
                l = 'input'
            layers_reps[l] = torch.Tensor([]).cuda()

        for im, im_path in tqdm(dl, desc=progress_label):
            im = im[0]
            if torch.cuda.is_available and const.DEBUG is False:
                im.cuda()
            try:
                im_key = os.path.basename(im_path[0])
                keys.append(im_key)
                point1_rep = re.get_layers_representation(im, im_key)
                for l in layers_reps:
                    t = torch.flatten(point1_rep[l].cuda()).unsqueeze(0)
                    layers_reps[l] = torch.cat([layers_reps[l], t])

            except:
                print(f'Error on {im_path}')

        layers_df = {}
        for l in layers_reps:
            relu = torch.nn.ReLU()
            print(layers_reps[l].shape)
            layers_reps[l] = layers_reps[l] / layers_reps[l].norm(dim=1, p=2)[:, None]
            distances = relu(1 - torch.mm(layers_reps[l], layers_reps[l].transpose(0, 1)))
            distances = distances.cpu().detach().numpy()
            layer_df = pd.DataFrame(distances, index=[keys], columns=keys)
            layers_df[l] = layer_df

        del re

        return layers_df

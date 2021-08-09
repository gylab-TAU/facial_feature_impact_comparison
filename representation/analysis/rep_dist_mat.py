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


class DistMatrixComparer(object):
    def __init__(self, reps_cache_path, image_loader, comparison_calc, get_layers_dict):
        self.__reps_cache_path = reps_cache_path
        self.__image_loader = image_loader
        self.__comparison_calc = comparison_calc
        self.__get_layers_dict = get_layers_dict

    def compare_pairs(self, model, dataset_dir, progress_label):
        dataset = ComparisonDataset(dataset_dir, self.__image_loader)
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
        layers_df = {}
        for l in self.__get_layers_dict(model).values():
            if l == 'model':
                l = 'input'
            layers_df[l] = pd.DataFrame()

        for im1, im1_path in tqdm(dl, desc=progress_label):
            im1 = im1[0]
            if torch.cuda.is_available and const.DEBUG is False:
                im1.cuda()
            try:
                row = {}
                for l in layers_df:
                    row[l] = {}
                im1_key = os.path.basename(im1_path[0])
                point1_rep = re.get_layers_representation(im1, im1_key)

                for im2, im2_path in dl:
                    im2 = im2[0]
                    im2_key = os.path.basename(im2_path[0])
                    if torch.cuda.is_available and const.DEBUG is False:
                        im2.cuda()
                    point2_rep = re.get_layers_representation(im2, im2_key+"comp")
                    for l in layers_df:
                        row[l][im2_key] = self.__comparison_calc.compare(point1_rep[l].cpu().numpy(), point2_rep[l].cpu().numpy())

                for l in layers_df:
                    p1_df_row = pd.DataFrame(row[l],
                        index=[im1_key])
                    layers_df[l] = layers_df[l].append(p1_df_row)

            except:
                print(f'Error on {im1_path}')

        del re

        return layers_df

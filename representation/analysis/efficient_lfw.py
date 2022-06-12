from typing import Dict

import torch
from representation.acquisition.representation_save_hook import FileSystemHook
from representation.acquisition.representation_extraction import RepresentationExtractor

import pandas as pd
import os.path
from tqdm import tqdm
import torch.utils.data as data
import const


class EfficientLFW(object):
    def __init__(self,
                 reps_cache_path: str,
                 dataset: data.DataLoader,
                 get_layers_dict,
                 verification_summary,
                 progress_label: str,
                 distance: str = 'cos'):
        self.__reps_cache_path = reps_cache_path
        self.__get_layers_dict = get_layers_dict
        self.__distance = distance
        self.__progress_label = progress_label
        self.__verification_summary = verification_summary
        self.__dataset = dataset

    def __get_img_reps(self, re, im: torch.Tensor, im_path: torch.Tensor):
        im = im.squeeze(dim=1)
        if torch.cuda.is_available and const.DEBUG is False:
            im.cuda()
            im_key = os.path.basename(im_path)
            point_rep = re.get_layers_representation(im, im_key)
            return point_rep

    def __summarize_layers(self,
                           layers_reps1: Dict[str, torch.Tensor],
                           layers_reps2: Dict[str, torch.Tensor],
                           labels: torch.Tensor) -> pd.DataFrame:
        """
        Calculate the distances between every pair of images (each row in a matrix is a representation of a single image)
        layers_reps1 - representations of images in column 1 in every wanted layer
        layers_reps2 - representations of images in column 2 in every wanted layer
        labels - same\diff {0, 1}
        """
        summary = {}
        layers = []
        for metric in self.__verification_summary.get_metric_name():
            summary[metric] = []

        # For every layer:
        for l in layers_reps1:
            layers.append(l)
            if self.__distance == 'cos':
                distances = 1 - torch.nn.CosineSimilarity(dim=1)(layers_reps1[l], layers_reps2[l])
            elif self.__distance.lower() == 'l2':
                distances = (layers_reps1[l] - layers_reps2[l]).norm(dim=1)


            layer_metric = self.__verification_summary.calc_performance(distances, labels)

            # Log the summarized verification performance of the network to return as a DataFrame
            # A row for each layer, and a column for each metric (like 'acc','best threshold')
            # For example:
            # |      | acc | threshold |
            # | Conv1| 0.65| 33.123124 |
            # ....
            for metric_name, measurement in zip(self.__verification_summary.get_metric_name(), layer_metric):
                summary[metric_name].append(measurement)
        return pd.DataFrame(summary, index=layers)


    def test_performance(self, model: torch.nn.Module) -> pd.DataFrame:
        re = RepresentationExtractor(model,
                                     self.__get_layers_dict(model),
                                     FileSystemHook(self.__get_layers_dict(model), self.__reps_cache_path))
        layers_reps1 = {}
        layers_reps2 = {}
        labels = []
        with torch.no_grad():
            for l in self.__get_layers_dict(model).values():
                if l == 'model':
                    l = 'input'
                layers_reps1[l] = []
                layers_reps2[l] = []

            # For every pair of images in the verification test:
            for im1, im2, im1_path, im2_path, label in tqdm(self.__dataset, desc=self.__progress_label):
                print(len(self.__dataset.dataset))
                labels.append(label)
                # Get the representations
                point_rep1 = self.__get_img_reps(re, im1, im1_path[0])
                point_rep2 = self.__get_img_reps(re, im2, im2_path[0])

                # If a matrix, flatten them (like after conv1 we get a tensor of (Height x Width x Depth),
                # flattening the tensor to a vector will enable us to use the same code for calculating the distances
                for l in layers_reps1:
                    t = torch.flatten(point_rep1[l].cuda(), start_dim=1)
                    layers_reps1[l].append(t)
                for l in layers_reps2:
                    t = torch.flatten(point_rep2[l].cuda(), start_dim=1)
                    layers_reps2[l].append(t)

            for l in layers_reps1:
                layers_reps1[l] = torch.cat(layers_reps1[l])
                layers_reps2[l] = torch.cat(layers_reps2[l])
            labels = torch.cat(labels).cuda()
            print(torch.sum(labels))
            del re

            return self.__summarize_layers(layers_reps1, layers_reps2, labels)



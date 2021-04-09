from representation.analysis.datapoints_rep_compare import DatapointsRepComparer
from representation.acquisition.representation_save_hook import FileSystemHook
from representation.acquisition.representation_extraction import RepresentationExtractor

import pandas as pd
import sys
import torch
from tqdm import tqdm


class PairsListComparer(object):
    def __init__(self, reps_cache_path, comparison_calc, get_layers_dict):
        self.__reps_cache_path = reps_cache_path
        self.__comparison_calc = comparison_calc
        self.__get_layers_dict = get_layers_dict

    def compare_lists(self, model, dataloader, progress_label):
        bad_indexes = []
        re = RepresentationExtractor(model,
                                     self.__get_layers_dict(model),
                                     FileSystemHook(self.__get_layers_dict(model), self.__reps_cache_path))

        comparisons_df = None

        for i, (im1, im1_keys, im2, im2_keys) in tqdm(enumerate(range(dataloader)), desc=progress_label):
            if torch.cuda.is_available():
                im1.cuda()
                im2.cuda()

            try:
                comp = DatapointsRepComparer(representation_extractor=re, comparison=self.__comparison_calc)
                comparison = comp.compare_datapoints(str(i) + '_im1', str(i) + '_im2', im1, im2)

                if type(comparison) == torch.Tensor:
                    comparison = comparison.cpu().numpy()

                comparison_df_row = pd.DataFrame(comparison, index=[im1_keys, im2_keys])

                if comparisons_df is None:
                    comparisons_df = comparison_df_row
                else:
                    comparisons_df = comparisons_df.append(comparison_df_row)
            except:
                print(sys.exc_info()[0])
                bad_indexes.append(i)

        del re

        return comparisons_df, bad_indexes

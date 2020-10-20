from representation.analysis.datapoints_rep_compare import DatapointsRepComparer
from representation.acquisition.representation_save_hook import FileSystemHook
from representation.acquisition.representation_extraction import RepresentationExtractor

import pandas as pd
import os.path
from tqdm import tqdm


class PairsListComparer(object):
    def __init__(self, reps_cache_path, image_loader, comparison_calc, get_layers_dict):
        self.__reps_cache_path = reps_cache_path
        self.__image_loader = image_loader
        self.__comparison_calc = comparison_calc
        self.__get_layers_dict = get_layers_dict

    def compare_pairs(self, model, dataset_dir, pairs_list, progress_label):
        bad_indexes = []
        re = RepresentationExtractor(model,
                                     self.__get_layers_dict(model),
                                     FileSystemHook(self.__get_layers_dict(model), self.__reps_cache_path))

        comparisons_df = None

        for i in tqdm(range(len(pairs_list)), desc=progress_label):
            try:
                im1_path = pairs_list[i][0]

                im2_path = pairs_list[i][1]
                if 'dd_ref' in im1_path and 'duchov' in im2_path:
                    x=1

                im1_path = os.path.join(dataset_dir, im1_path)
                im2_path = os.path.join(dataset_dir, im2_path)

                im1 = self.__image_loader.load_image(im1_path)
                im2 = self.__image_loader.load_image(im2_path)

                im1_key = os.path.basename(im1_path)
                im2_key = os.path.basename(im2_path)


                comp = DatapointsRepComparer(representation_extractor=re, comparison=self.__comparison_calc)
                comparison = comp.compare_datapoints(im1_key, im2_key, im1, im2)

                comparison_df_row = pd.DataFrame({(pairs_list[i][0], pairs_list[i][1]): comparison}).transpose()
                comparison_df_row['type'] = progress_label

                if comparisons_df is None:
                    comparisons_df = comparison_df_row
                else:
                    comparisons_df.append(comparison_df_row)
            except:
                print(f'Error on {im1_path}, {im2_path}')
                bad_indexes.append(i)

        del re

        return comparisons_df, bad_indexes

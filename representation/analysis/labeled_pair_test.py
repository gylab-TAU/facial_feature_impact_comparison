from representation.analysis.datapoints_rep_compare import DatapointsRepComparer
from representation.acquisition.representation_save_hook import FileSystemHook
from representation.acquisition.representation_extraction import RepresentationExtractor

import pandas as pd
import const
import sys
import torch
import tqdm

class PerformanceTester(object):
    def __init__(self, representation_performance_analyzer, dataloader, comparison_calc,
                 progress_label: str):
        self.__rep_perf_analyzer = representation_performance_analyzer
        self.dataloader = dataloader
        self.comparison_calc = comparison_calc
        self.progress_label = progress_label

    def test_performance(self, model):
        working_labels_list = []
        re = RepresentationExtractor(model,
                                     self.__get_layers_dict(model),
                                     FileSystemHook(self.__get_layers_dict(model), self.__reps_cache_path))

        comparisons_df = None

        for i, (im1, im1_keys, im2, im2_keys, pair_labels) in tqdm(enumerate(range(self.dataloader)), desc=self.progress_label):
            working_labels_list = working_labels_list + pair_labels
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
        del re

        # Then, per layer we calculate the layer's accuracies
        columns = [const.LAYER] + self.__rep_perf_analyzer.get_metric_name()
        reduced_performance_df = pd.DataFrame(columns=columns)

        reduced_performance_df[const.LAYER] = comparisons_df.columns
        reduced_performance_df = reduced_performance_df.set_index(const.LAYER)
        for key in comparisons_df.columns:
            reduced_performance_df.loc[key] = self.__rep_perf_analyzer.calc_performance(comparisons_df[key], working_labels_list)




class MultiListComparer(object):
    def __init__(self, dataloaders: dict, pairs_list_comparison):
        self.__dataloaders = dataloaders
        self.__pairs_list_comparison = pairs_list_comparison

    def compare_lists(self, model):
        comparisons_df = None
        for pairs_type in self.__dataloaders:
            type_comparisons_df, _ = self.__pairs_list_comparison.compare_pairs(model,
                                                                             self.__dataloaders,
                                                                             pairs_type)
            type_comparisons_df['type'] = pairs_type
            if comparisons_df is None:
                comparisons_df = type_comparisons_df
            else:
                comparisons_df = comparisons_df.append(type_comparisons_df)

        return comparisons_df

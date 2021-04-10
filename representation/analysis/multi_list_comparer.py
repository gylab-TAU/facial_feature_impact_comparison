
class MultiListComparer(object):
    def __init__(self, pairs_types_to_lists, pairs_types_to_dir, pairs_list_comparison):
        self.__pairs_types_to_lists = pairs_types_to_lists
        self.__pairs_types_to_dir = pairs_types_to_dir
        self.__pairs_list_comparison = pairs_list_comparison

    def compare_lists(self, model):
        comparisons_df = None
        for pairs_type in self.__pairs_types_to_lists:
            type_comparisons_df, _ = self.__pairs_list_comparison.compare_pairs(model,
                                                                             self.__pairs_types_to_dir[pairs_type],
                                                                             self.__pairs_types_to_lists[pairs_type],
                                                                             pairs_type)
            type_comparisons_df['type'] = pairs_type
            if comparisons_df is None:
                comparisons_df = type_comparisons_df
            else:
                comparisons_df = comparisons_df.append(type_comparisons_df)

        return comparisons_df


class PerformanceTester(object):
    def __init__(self, representation_performance_analyzer, pairs_list_comparison):
        self.__rep_perf_analyzer = representation_performance_analyzer
        self.__pairs_list_comparison = pairs_list_comparison

    def test_performance(self, model, dataset_dir, pairs_list, labels_list, progress_label):
        # First we get all pair comparisons with the list of labels
        working_labels_list = labels_list.copy()
        comparisons_by_layers, bad_indexes = self.__pairs_list_comparison.compare_pairs(model, dataset_dir, pairs_list, progress_label)

        for i in bad_indexes:
            del working_labels_list[i]

        # Then, per layer we calculate the layer's accuracies
        reduced_performance = {}
        for key in comparisons_by_layers:
            reduced_performance[key] = self.__rep_perf_analyzer.calc_performance(comparisons_by_layers[key], working_labels_list)

        return reduced_performance


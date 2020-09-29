
class PerformanceTester(object):
    def __init__(self, representation_performance_analyzer, pairs_list_comparison):
        self.__rep_perf_analyzer = representation_performance_analyzer
        self.__pairs_list_comparison = pairs_list_comparison

    def test_performance(self, model, dataset_dir, pairs_list, labels_list, progress_label):
        # First we get all pair comparisons with the list of labels
        comparisons_by_layers = self.__pairs_list_comparison.compare_pairs(model, dataset_dir, pairs_list, progress_label)

        # Then, per layer we calculate the layer's accuracies
        reduced_performance = {}
        for key in comparisons_by_layers:
            reduced_performance[key] = self.__rep_perf_analyzer.calc_performance(comparisons_by_layers[key], labels_list)

        return reduced_performance


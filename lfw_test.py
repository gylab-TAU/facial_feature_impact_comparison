
class LFWTester(object):
    def __init__(self, pairs_list: list, labels_list: list,
                 lfw_dir: str, progress_label: str, performance_tester):
        self.__pairs_list = pairs_list
        self.__labels_list = labels_list
        self.__lfw_dir = lfw_dir
        self.__progress_label = progress_label
        self.__performance_tester = performance_tester

    def test_performance(self, model):
        return self.__performance_tester.test_performance(model,
                                                          self.__lfw_dir,
                                                          self.__pairs_list,
                                                          self.__labels_list,
                                                          self.__progress_label)


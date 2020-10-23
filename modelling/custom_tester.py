import torch


class CustomTester(object):
    def __init__(self, model: torch.nn.modules.Module, performance_tester, performance_logger):
        self.model = model
        self.__performance_tester = performance_tester
        self.__performance_logger = performance_logger

    def run(self, epoch):
        performance_df = self.__performance_tester.test_performance(self.model)
        self.__performance_logger.log_performance(epoch, performance_df)
        highest_acc = performance_df['acc@1'].max()
        highest_acc_layer = performance_df['acc@1'].idxmax()
        highest_acc_thresh = performance_df.loc[highest_acc_layer]['threshold']
        return highest_acc, highest_acc_thresh, highest_acc_layer

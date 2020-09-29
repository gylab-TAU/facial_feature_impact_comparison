import torch.optim as optim


class LRSchedulerInitializer(object):
    def __init__(self, step, gamma):
        self.__step = step
        self.__gamma = gamma

    def get_scheduler(self, scheduler_name, optimizer, last_epoch):
        return optim.lr_scheduler.StepLR(optimizer, self.__step, last_epoch)
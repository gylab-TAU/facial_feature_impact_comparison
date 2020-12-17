import torch.optim as optim


class GenericLRSchedulerInitializer(object):
    """
    Generic LR scheduler factory, using reflection to create the scheduler
    """

    def get_scheduler(self, scheduler_name: str, optimizer: optim.Optimizer, params: {}, last_epoch: int = -1):
        """
        Producing a lr scheduler given the name and params (using reflection)
        :param scheduler_name: the string name of the scheduler as it appears in torch.optim.lr_scheduler
        :param optimizer: the optimizer we adapt
        :param params: the scheduler specific params
        :param last_epoch: the last epoch the model was trained. default -1 if wasn't trained yet.
        :return: the scheduler object
        """
        return optim.lr_scheduler.__dict__[scheduler_name](optimizer=optimizer, last_epoch=last_epoch, **params)
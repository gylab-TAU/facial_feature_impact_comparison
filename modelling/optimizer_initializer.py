import torch


class SGDOptimizerInitializer(object):
    def __init__(self, lr=0.01, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay=weight_decay
        self.dampening=dampening
        self.neseterov=nesterov

    def get_optimizer(self, optimizer_name, model):
        return torch.optim.SGD(model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay,
                               dampening=self.dampening,
                               nesterov=self.neseterov)
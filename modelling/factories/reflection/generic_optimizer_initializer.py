import torch


class GenericOptimizerInitializer(object):
    """
    a torch.optim.Optimizer factory, creating an optimizer using reflection
    """

    def get_optimizer(self, optimizer_name: str, model: torch.nn.Module, params: {}) -> torch.optim.Optimizer:
        """
        Creating a torch.optim.Optimizer using reflection
        :param optimizer_name: the optimizer name as it appears in torch.optim
        :param model: the model instance we wish to optimize
        :param params: a dictionary with the optimizer's specific parameters
        :return: the instantiated optimizer
        """
        return torch.optim.__dict__[optimizer_name](model.parameters(), **params)

import torch
import torch.nn
import const
from modelling import loss


class CustomCriterionInitializer(object):
    """
    factory using reflection to initialize a criterion
    first checking for custom loss functions, then trying for a pytorch implementation.
    """
    def get_criterion(self, criterion_name: str, parameters: {} = {}):
        """
        Initializing a criterion using reflection and using cuda (if GPU is available)
        :param criterion_name: the criterion name as it appears in torch.nn or in modelling.loss
        :param parameters: a dictionary containing the parameters for the criterion. if empty, uses torch's default values
        :return: the initialized criterion - set on available device
        """
        if criterion_name in loss.__dict__:
            criterion = loss.__dict__[criterion_name](**parameters)
        else:
            criterion = torch.nn.__dict__[criterion_name](**parameters)

        if torch.cuda.is_available() and const.DEBUG is False:
            criterion = criterion.cuda()
        return criterion

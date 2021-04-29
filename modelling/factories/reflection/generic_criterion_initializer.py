import torch
import torch.nn


class GenericCriterionInitializer(object):
    """
    factory using reflection to initialize a criterion
    """
    def get_criterion(self, criterion_name: str, parameters: {} = {}):
        """
        Initializing a criterion using reflection and using cuda (if GPU is available)
        :param criterion_name: the criterion name as it appears in torch.nn
        :param parameters: a dictionary containing the parameters for the criterion. if empty, uses torch's default values
        :return: the initialized criterion - set on available device
        """
        criterion = torch.nn.__dict__[criterion_name](**parameters)
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        return criterion

import torch
import torch.nn
import const


def cos_dist(x: torch.Tensor, y: torch.Tensor):
    x = x / x.norm(dim=1, p=2)[:, None]
    y = y / y.norm(dim=1, p=2)[:, None]
    dists = torch.diagonal(1 - (x @ y.transpose(0, 1)))
    return dists


class TripletCriterionInitializer(object):
    """
    Factory to create triplet loss function
    """

    def __init__(self, inner_factory):
        self.inner_factory = inner_factory

    def get_criterion(self, criterion_name: str, parameters: {} = {}):
        """
        Initializing a criterion using reflection and using cuda (if GPU is available)
        :param criterion_name: the criterion name as it appears in torch.nn, or 'triplet'
        :param parameters: a dictionary containing the parameters for the criterion. if empty, uses torch's default values
        if criterion_name is triplet, if 'distance' in parameters keys,
        :return: the initialized criterion - set on available device
        """
        if criterion_name.lower() == 'triplet':
            print(parameters)
            if 'distance' in parameters:
                dist = parameters['distance']
                del parameters['distance']
                if dist == 'cos':
                    if 'margin' in parameters:
                        criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=cos_dist, margin=parameters['margin'])
                    else:
                        criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=cos_dist)
                else:
                    criterion = torch.nn.TripletMarginLoss(**parameters)
            else:
                criterion = torch.nn.TripletMarginLoss(**parameters)
        else:
            criterion = self.inner_factory.get_criterion(criterion_name, parameters)
        if torch.cuda.is_available() and const.DEBUG is False:
            criterion = criterion.cuda()
        return criterion

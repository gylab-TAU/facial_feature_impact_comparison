import torch
from modelling.factories.reflection.generic_criterion_initializer import GenericCriterionInitializer
from modelling.loss.arcface import ArcFace
import const


class ArcFaceCriterionInitializer(GenericCriterionInitializer):
    """
    A criterion initializer, creating both standard pytorch criteria, as well as ArcFace loss
    """
    def __init__(self):
        super(ArcFaceCriterionInitializer, self).__init__()

    def get_criterion(self, criterion_name: str, parameters: dict = {}):
        if criterion_name.lower() == 'arcface':
            criterion = ArcFace(**parameters)
        else:
            criterion = super(ArcFaceCriterionInitializer, self).get_criterion(criterion_name, **parameters)
        if torch.cuda.is_available() and const.DEBUG is False:
            criterion = criterion.cuda()
        return criterion

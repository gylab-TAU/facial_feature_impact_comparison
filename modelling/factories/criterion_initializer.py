import torch
import torch.nn
import const


class CrossEntropyCriterionInitializer(object):
    def get_criterion(self, criterion_name):
        criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available() and const.DEBUG is False:
            criterion = criterion.cuda()
        return criterion

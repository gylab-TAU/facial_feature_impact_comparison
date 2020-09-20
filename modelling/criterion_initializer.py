import torch
import torch.nn


class CrossEntropyCriterionInitializer(object):
    def get_criterion(self, criterion_name):
        criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        return criterion

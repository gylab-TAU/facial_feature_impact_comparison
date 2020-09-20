import torchvision
import torchvision.models as models
import torch
from .trainer import Trainer


class ModelInitializer(object):
    """An initializer object retrieving a model customized for the single machine we are using."""
    def __init__(self,feature_parallelized_archs:list =[]):
        """

        :param feature_parallelized_archs: list of architectures whose feature can be parallelized over GPU.
        """
        self.feature_parallelized_archs = feature_parallelized_archs

    def get_model(self, arch:str, is_pretrained:bool, num_classes:int):
        """
        Returns an initialized model object, set to work with machine's resources
        :param arch: name of architecure to use
        :param is_pretrained: Should we use a torch pretrained model
        :param num_classes: Number of classes the model should know
        :return: The initialized model
        """
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=is_pretrained, num_classes=num_classes)
        if torch.cuda.is_available():

            # DataParallel will divide and allocate batch_size to all available GPUs
            if arch in self.feature_parallelized_archs:
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        else:
            print('Using CPU, run times may be long.')

        return model


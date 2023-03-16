import torchvision.models as models
import torch
import const

from .model_initializer import ModelInitializer


class TripletModelInitializer(ModelInitializer):
    """An initializer object retrieving a model customized for the single machine we are using."""
    def __init__(self, feature_parallelized_archs: list = []):
        super(TripletModelInitializer, self).__init__(feature_parallelized_archs)

    def get_model(self, arch: str, is_pretrained: bool, num_classes: int, triplet_train: bool) -> torch.nn.Module:
        """
        Returns an initialized model object, set to work with machine's resources
        :param arch: name of architecure to use
        :param is_pretrained: Should we use a torch pretrained model
        :param num_classes: Number of classes the model should know
        :return: The initialized model
        """
        print("=> creating model '{}'".format(arch))
        output_dim = num_classes
        if is_pretrained:
            num_classes = 1000
        elif triplet_train:
            num_classes = 2
        model = super(TripletModelInitializer, self).get_model(arch, is_pretrained, num_classes)
        if arch == 'resnet34':
            model.fc = torch.nn.Linear(model.fc.in_features, output_dim)
            model.fc.cuda()
            model.cuda()

        return model


from torch import nn
import torch
import const
import modelling.models.verification_model
from modelling.models.reflection_verification_model import ReflectionVerificationModel


class VerificationModelInitializer(object):
    """An initializer object retrieving a model customized for the single machine we are using."""
    def __init__(self, inner_initializer, layers_extractor, reps_cache_path: str, verification_score: nn.Module):
        """

        :param inner_initializer: a factory creating inner models
        :param layers_extractor: the layer extractor to use
        :param reps_cache_path: where to save the representations (temporarily)
        :param verification_score: The score function to use
        """
        self.inner_initializer = inner_initializer
        self.layers_extractor = layers_extractor
        self.reps_cache_path = reps_cache_path
        self.verification_score = verification_score

    def get_model(self, arch: str,
                  is_pretrained: bool,
                  num_classes: int) -> modelling.models.verification_model.VerificationModel:
        """
        Returns an initialized model object, set to work with machine's resources
        :param arch: name of architecture to use
        :param is_pretrained: Should we use a torch pretrained model
        :param num_classes: Number of classes the model should know
        :param verification_layers: String representing which layers to use for deep representations
        :return: The initialized model
        """
        print("=> creating model '{}'".format(arch))
        inner_model = self.inner_initializer.get_model(arch, is_pretrained, num_classes)
        model = ReflectionVerificationModel(
            inner_model,
            self.reps_cache_path,
            self.verification_score,
            self.layers_extractor)

        if torch.cuda.is_available() and const.DEBUG is False:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if arch in self.feature_parallelized_archs:
                model.features = torch.nn.DataParallel(model.features)
                if const.DEBUG is False:
                    model.cuda()
            else:
                model = torch.nn.DataParallel(model)
                if const.DEBUG is False:
                    model.cuda()
        else:
            print('Using CPU, run times may be long.')

        return model


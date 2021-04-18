from modelling.models.arcvgg import ArcVGG
from modelling.factories.model_initializer import ModelInitializer
import torch


class ArcFaceModelInitializer(ModelInitializer):
    """
    A factory for creating both standard models, as well as Arccos models
    """
    def __init__(self, feature_parallelized_archs: list = []):
        super(ArcFaceModelInitializer, self).__init__(feature_parallelized_archs)

    def get_model(self, arch: str, is_pretrained: bool, num_classes: int, arc: bool = False) -> torch.nn.Module:
        if arc:
            if arch.startswith('vgg'):
                model = ArcVGG(arch, num_classes)
                if torch.cuda.is_available():
                    # DataParallel will divide and allocate batch_size to all available GPUs
                    if arch in self.feature_parallelized_archs:
                        model.features = torch.nn.DataParallel(model.features)
                        model.cuda()
                    else:
                        model = torch.nn.DataParallel(model).cuda()
        else:
            model = super(ArcFaceModelInitializer, self).get_model(arch, is_pretrained, num_classes)
        return model

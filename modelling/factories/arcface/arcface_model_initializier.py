from modelling.models.arcvgg import ArcVGG
import modelling.factories
import torch


class ArcFaceModelInitializer(modelling.factories.ModelInitializer):
    """
    A factory for creating both standard models, as well as Arccos models
    """
    def __init__(self, feature_parallelized_archs: list = []):
        super(ArcFaceModelInitializer, self).__init__(feature_parallelized_archs)

    def get_model(self, arch: str, is_pretrained: bool, num_classes: int, arc: bool = False) -> torch.nn.Module:
        if arc:
            if arch.startswith('vgg'):
                return ArcVGG(arch, num_classes)
        return super(ArcFaceModelInitializer, self)(arch, is_pretrained, num_classes)

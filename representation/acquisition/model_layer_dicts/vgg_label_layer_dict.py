from torch import nn


class VggLabelLayerDict(object):
    """
    Given a text label (for VGG16 only!) get the specific model layer
    """
    def __init__(self, layer_label):
        index_dict = {
            'model': 0,
            'conv1': 7,
            'conv2': 12,
            'conv3': 19,
            'conv4': 26,
            'conv5': 33,
            'fc6': 37,
            'fc7': 40,
            'fc8': 42}
        self.layer_label = layer_label
        self.idx = index_dict[layer_label]

    def __call__(self, model: nn.Module):
        layers_dict = {}
        for i, layer in enumerate(model.modules()):
            if i == self.idx:
                layers_dict[layer] = self.layer_label

        return layers_dict

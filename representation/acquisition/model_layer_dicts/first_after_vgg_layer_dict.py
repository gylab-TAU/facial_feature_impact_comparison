from torch import nn


class FirstAfterVGGLayerDict(object):
    """
    Given a text label (for VGG16 only!) get the specific model layer
    """
    def __init__(self, layer_label):
        index_dict = {
            'first_after_conv1': 4,
            'first_after_conv2': 9,
            'first_after_conv3': 14,
            'first_after_conv4': 21,
            'first_after_conv5': 28,
            'first_after_fc6': 37,
            'first_after_fc7': 40,
            'first_after_fc8': 42}
        self.layer_label = layer_label
        self.idx = index_dict[layer_label]

    def __call__(self, model: nn.Module):
        layers_dict = {}
        for i, layer in enumerate(model.modules()):
            if i == self.idx:
                layers_dict[layer] = self.layer_label

        return layers_dict

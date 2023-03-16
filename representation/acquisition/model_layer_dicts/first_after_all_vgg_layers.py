from torch import nn


class FirstAfterAllVGGLayerDict(object):
    """
    Given a text label (for VGG16 only!) get the specific model layer
    """
    def __call__(self, model: nn.Module):
        index_dict = {
            4: 'first_after_conv1',
            9: 'first_after_conv2',
            14: 'first_after_conv3',
            21: 'first_after_conv4',
            28: 'first_after_conv5',
            37: 'first_after_fc6',
            40: 'first_after_fc7',
            42: 'first_after_fc8'}
        layers_dict = {}
        for i, layer in enumerate(model.modules()):
            if i in index_dict:
                layers_dict[layer] = index_dict[i]
        print(layers_dict)
        return layers_dict

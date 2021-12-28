from torch import nn


class Second2Last(object):
    def __call__(self, model: nn.Module):
        l = 0
        layers_dict = {}

        for _ in model.modules():
            l += 1

        for i, layer in enumerate(model.modules()):
            if i == (l - 1):
                layers_dict[layer] = i

        return layers_dict

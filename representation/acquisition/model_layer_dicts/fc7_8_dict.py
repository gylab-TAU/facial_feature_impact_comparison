import torch


class Fc78Dict(object):
    def __call__(self, model):
        layers_list = []
        for i, layer in enumerate(model.modules()):
            if type(layer) != torch.nn.modules.container.Sequential:
                layers_list.append(layer)
        layers_dict = {}
        layers_dict[layers_list[-1]] = len(layers_list) - 1
        # Get the second to last linear layer
        for i in range(len(layers_list) - 2, -1, -1):
            layers_dict[layers_list[i]] = i
            if type(layers_list[i]) == torch.nn.modules.linear.Linear:
                return layers_dict
import torch


def get_model_layers_dict(model: torch.nn.Module):
    """
    Given a model, retrieves a dictionary containing only the representation layer with it's index
    :param model: The model whose layers we want to use
    :return: dictionary mapping { actual_model_layer_object: index_in_the_model }
    """

    layers_list = []
    for i, layer in enumerate(model.modules()):
        if type(layer) != torch.nn.modules.container.Sequential:
            layers_list.append(layer)
    layers_dict = {}
    # Get the second to last linear layer
    for i in range(len(layers_list) - 2, -1, -1):
        layers_dict[layers_list[i]] = i
        if type(layers_list[i]) == torch.nn.modules.linear.Linear:
            return layers_dict

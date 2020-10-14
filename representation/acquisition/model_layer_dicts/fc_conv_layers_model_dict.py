import torch


def get_model_layers_dict(model: torch.nn.Module):
    """
    Given a model, retrieves a dictionary containing only the FC layers and the CONV layers with it's index
    :param model: The model whose layers we want to use
    :return: dictionary mapping { actual_model_layer_object: index_in_the_model }
    """

    layers_dict = {}
    start_index = 2
    for i, layer in enumerate(model.modules()):
        if type(layer) == torch.nn.Conv2d or type(layer) == torch.nn.modules.linear.Linear:
            layers_dict[layer] = start_index
            start_index += 1

    return layers_dict

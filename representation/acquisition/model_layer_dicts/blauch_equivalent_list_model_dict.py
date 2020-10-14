import torch


def get_model_layers_dict(model: torch.nn.Module):
    """
    Given a model, retrieves a dictionary containing only the FC layers and the CONV layers with it's index
    :param model: The model whose layers we want to use
    :return: dictionary mapping { actual_model_layer_object: index_in_the_model }
    """
    index_list = [0, 7, 12, 19, 26, 33, 37, 40, 42]
    layers_dict = {}
    for i, layer in enumerate(model.modules()):
        if i in index_list:
            layers_dict[layer] = i

    return layers_dict

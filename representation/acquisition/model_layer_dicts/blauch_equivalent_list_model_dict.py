import torch


def get_model_layers_dict(model: torch.nn.Module):
    """
    Given a model, retrieves a dictionary containing only the FC layers and the CONV layers with it's index
    :param model: The model whose layers we want to use
    :return: dictionary mapping { actual_model_layer_object: index_in_the_model }
    """
    index_dict = {0: 'model', 7: 'conv1', 12: 'conv2', 19: 'conv3', 26: 'conv4', 33: 'conv5', 37: 'fc6', 40: 'fc7', 42: 'fc8'}
    layers_dict = {}
    for i, layer in enumerate(model.modules()):
        if i in index_dict:
            layers_dict[layer] = index_dict[i]

    return layers_dict

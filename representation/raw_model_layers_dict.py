import torch


def get_model_layers_dict(model: torch.nn.Module):
    """
    Given a model, retrieves a dictionary matching every layer of it with it's index (relative place throughout the model
    Doesn't include the model itself
    Doesn't include Sequential type layers, however does include the output of every layer in the sequence
    :param model: The model whose layers we want to use
    :return: dictionary mappint { actual_model_layer_object: index_in_the_model }
    """
    layers_dict = {}
    for i, layer in enumerate(model.modules()):
        if type(layer) != torch.nn.modules.container.Sequential:
            layers_dict[layer] = i
    return layers_dict

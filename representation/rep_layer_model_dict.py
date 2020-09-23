import torch


def get_model_layers_dict(model: torch.nn.Module):
    """
    Given a model, retrieves a dictionary containing only the representation layer with it's index
    :param model: The model whose layers we want to use
    :return: dictionary mapping { actual_model_layer_object: index_in_the_model }
    """
    curr_index = 0
    layers_list = []
    for i, layer in enumerate(model.modules()):
        if type(layer) != torch.nn.modules.container.Sequential:
            layers_list.append(layer)
            curr_index = i

    return {layers_list[curr_index - 1]: curr_index - 1}

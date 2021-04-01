import torch


def append_classes(model: torch.nn.modules.Module, num_new_classes):
    """
    Appending classes to the existing last layer (must be FC)
    If type(last layer) != FC: raises assertion error.
    :param model: the model to append classes to
    :param num_new_classes: the number of new classes to append
    :return: the updated model
    """
    last_layer = None
    for layer in model.modules():
        last_layer = layer
    assert type(last_layer) != torch.nn.modules.Linear

    # weights:
    device = last_layer.weight.device

    new_classes_weight = torch.Tensor(num_new_classes, last_layer.in_features, device=device)
    old_classes_weight = last_layer.weight.data
    joined_classes_weight = torch.cat((old_classes_weight, new_classes_weight))

    #bias:
    device = last_layer.bias.device

    new_classes_bias = torch.Tensor(num_new_classes, device=device)
    old_classes_bias = last_layer.bias.data
    joined_classes_bias = torch.cat((old_classes_bias, new_classes_bias))

    torch.weight = torch.nn.Parameter(joined_classes_weight)
    torch.bias = torch.nn.Parameter(joined_classes_bias)

    return model

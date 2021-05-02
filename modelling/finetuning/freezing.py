import torch


def freeze_layers(model: torch.nn.modules.Module, freeze_end_layer_index: int):
    """
    setting all parameters requires_grad=False for all parameters, up to freeze_end_layer_index (exclusive, zero based)
    :param model: the model to freeze
    :param freeze_end_layer_index: the index of the first layer that shouldn't be frozen (zero based)
    :return: the frozen model
    """
    for i, mod in enumerate(model.modules()):
        if i < freeze_end_layer_index:
            for param in mod.parameters():
                param.requires_grad = False
        else:
            for param in mod.parameters():
                param.requires_grad = True
    return model

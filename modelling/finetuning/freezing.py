import torch


def freeze_layers(model: torch.nn.modules.Module, freeze_end_layer_index: int):
    """
    setting all parameters requires_grad=False for all parameters, up to freeze_end_layer_index (exclusive, zero based)
    :param model: the model to freeze
    :param freeze_end_layer_index: the index of the first layer that shouldn't be frozen (zero based)
    :return: the frozen model
    """
    layers = 0
    for i, mod in enumerate(model.modules()):
        if i < freeze_end_layer_index:
            mod.requires_grad_(False)
            for param in mod.parameters():
                param.requires_grad = False
        else:
            mod.requires_grad_(True)
            for param in mod.parameters():
                param.requires_grad = True
        layers = i
    for i, mod in enumerate(model.modules()):
        for param in mod.parameters():
            if param.requires_grad:
                # print(f"Training layer: idx={i}, type={mod}, from_end={layers-i}")
                break
            else:
                break
    return model

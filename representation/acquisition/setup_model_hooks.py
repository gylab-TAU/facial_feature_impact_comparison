import torch


def setup_model_hooks(model: torch.nn.Module, hook, layers_dict):
    """
    setting up the forward hooks of a given model for specified layers
    :param model: The model we hook
    :param hook: The hook object, containing save method, and save_model_io (for saving both input and output of full model)
    :param layers_dict: { layer_object : layer_index_in_the_model}
    :return: the hooked model
    """
    removeable_handles_list = []
    for layer in model.modules():
        if layer in layers_dict:
            if type(layer) == type(model):
                removeable_handles_list.append(layer.register_forward_hook(lambda model_layer, inp, output: hook.save_model_io(model_layer, inp, output)))
            else:
                removeable_handles_list.append(layer.register_forward_hook(lambda model_layer, inp, output: hook.save(model_layer, inp, output)))
    return removeable_handles_list

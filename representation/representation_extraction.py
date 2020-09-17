from representation.setup_model_hooks import setup_model_hooks
import torch


class RepresentationExtractor(object):
    """
    A wrapper for a model, given a tensor, we extract the representation throughout the model, and return the reps.
    """
    def __init__(self, model: torch.nn.Module, model_layer_dict, representation_hook):
        self.model = model
        self.model_layers_dict = model_layer_dict
        self.representation_hook = representation_hook
        setup_model_hooks(model, representation_hook, model_layer_dict)

    def get_layers_representation(self, data_point: torch.Tensor, data_point_key):
        self.save_layers_representation(data_point, data_point_key)
        representations = self.representation_hook.load()
        return representations

    def save_layers_representation(self, data_point, data_point_key):
        self.representation_hook.set_data_point_key(data_point_key)
        self.model(data_point)

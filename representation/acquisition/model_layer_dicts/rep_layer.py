from representation.acquisition.model_layer_dicts import rep_layer_model_dict


class RepLayerModelDict(object):
    def __call__(self, model):
        return rep_layer_model_dict.get_model_layers_dict(model)
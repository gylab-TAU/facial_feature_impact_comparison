import representation.acquisition.model_layer_dicts.blauch_equivalent_list_model_dict as blauch_equivalent_list_model_dict


class BlauchEquivalentExtractor(object):
    def __call__(self, model):
        return blauch_equivalent_list_model_dict.get_model_layers_dict(model)
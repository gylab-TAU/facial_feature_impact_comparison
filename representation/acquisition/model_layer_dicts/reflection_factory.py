from representation.acquisition.model_layer_dicts import blauch_eq_extract
from representation.acquisition.model_layer_dicts import prior_to_last
from representation.acquisition.model_layer_dicts import rep_layer
from representation.acquisition.model_layer_dicts import fc7_8_dict
from representation.acquisition.model_layer_dicts import vgg_label_layer_dict
from representation.acquisition.model_layer_dicts import second_2_last


class ReflectionFactory(object):
    def get_dict_extractor(self, type: str):
        if type == 'PriorToLastDictExtractor':
            return prior_to_last.PriorToLastDictExtractor()
        if type == 'BlauchEquivalentExtractor':
            return blauch_eq_extract.BlauchEquivalentExtractor()
        if type == 'RepLayer':
            return rep_layer.RepLayerModelDict()
        if type == 'Fc78Dict':
            return fc7_8_dict.Fc78Dict()
        if type in ['model', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
            return vgg_label_layer_dict.VggLabelLayerDict(type)
        if type == 'Second2Last':
            return second_2_last.Second2Last()

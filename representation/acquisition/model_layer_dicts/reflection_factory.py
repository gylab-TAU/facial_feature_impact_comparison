from representation.acquisition.model_layer_dicts import blauch_eq_extract
from representation.acquisition.model_layer_dicts import prior_to_last
from representation.acquisition.model_layer_dicts import rep_layer
from representation.acquisition.model_layer_dicts import fc7_8_dict
from representation.acquisition.model_layer_dicts import vgg_label_layer_dict
from representation.acquisition.model_layer_dicts import second_2_last
from representation.acquisition.model_layer_dicts import first_after_all_vgg_layers
from representation.acquisition.model_layer_dicts import first_after_vgg_layer_dict


class ReflectionFactory(object):
    def get_dict_extractor(self, t: str):
        if t == 'PriorToLastDictExtractor':
            return prior_to_last.PriorToLastDictExtractor()
        if t == 'BlauchEquivalentExtractor':
            return blauch_eq_extract.BlauchEquivalentExtractor()
        if t == 'RepLayer':
            return rep_layer.RepLayerModelDict()
        if t == 'Fc78Dict':
            return fc7_8_dict.Fc78Dict()
        if t in ['model', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
            return vgg_label_layer_dict.VggLabelLayerDict(t)
        if t == 'Second2Last':
            return second_2_last.Second2Last()
        if t == 'FirstAfterAllVGGLayerDict':
            return first_after_all_vgg_layers.FirstAfterAllVGGLayerDict()
        if t in ['first_after_conv1', 'first_after_conv2', 'first_after_conv3', 'first_after_conv4', 'first_after_conv5', 'first_after_fc6', 'first_after_fc7', 'first_after_fc8']:
            return first_after_vgg_layer_dict.FirstAfterVGGLayerDict(t)



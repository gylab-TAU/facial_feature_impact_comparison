from representation.acquisition.model_layer_dicts import blauch_eq_extract
from representation.acquisition.model_layer_dicts import prior_to_last
from representation.acquisition.model_layer_dicts import rep_layer


class ReflectionFactory(object):
    def get_dict_extractor(self, type):
        if type == 'PriorToLastDictExtractor':
            return prior_to_last.PriorToLastDictExtractor()
        if type == 'BlauchEquivalentExtractor':
            return blauch_eq_extract.BlauchEquivalentExtractor()
        if type == 'RepLayer':
            return rep_layer.RepLayerModelDict()
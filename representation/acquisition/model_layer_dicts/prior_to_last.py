import torch

class PriorToLastDictExtractor(object):
    def __call__(self, model, *args, **kwargs):
        layers_dict = []
        for i, layer in enumerate(model.modules()):
            if type(layer) != torch.nn.modules.container.Sequential and type(layer) != torch.nn.parallel.DataParallel:
                layers_dict.append(layer)
        return {layers_dict[-2]: len(layers_dict)-2}
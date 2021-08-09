import pickle
import glob
import os
import torch.nn as nn
import torch
import const


class FileSystemHook(object):
    """
    A representation hook class, pickling the reps to the FS
    """
    def __init__(self, layers_dict, root_dir, delete_after_load=True):
        self.__layers_dict = layers_dict
        self.__root_dir = root_dir
        self.__filename = None
        self.__delete_after_load = delete_after_load
        self.__transform = nn.ReLU() # nn.Softmax(dim=1)
        if torch.cuda.is_available() and const.DEBUG is False:
            self.__transform.cuda()

    def set_data_point_key(self, key):
        """
        Setting the datapoint's key (for saving the reps for a customized location, later making it easier to load the data.
        :param key: The key to the data point
        :return: None
        """
        self.__filename = key

    def save_model_io(self, layer, input, output):
        """
        A hook, pickling both the model's input and output.
        :param layer: the model 'layer'
        :param input: the layer's input
        :param output: the layer's output
        :return: None
        """
        input_path = os.path.join(self.__get_data_item_dir(), 'input.pkl')
        output_path = os.path.join(self.__get_data_item_dir(), 'output.pkl')
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'rep': self.__transform(output).clone().detach().cpu(),
                'layer': 'output'
            }, f)

        with open(input_path, 'wb') as f:
            pickle.dump({
                'rep': input[0].clone().detach().cpu(),
                'layer': 'input'
            }, f)


    def save(self, layer, inp, output):
        """
        Hook pickling a layer's output representation
        :param layer: the layer (used to define where to pickle the rep)
        :param inp: the layer's input (ignored)
        :param output: the layer's output (which we save)
        :return: None
        """
        with open(self.__get_rep_path(layer), 'wb') as f:
            pickle.dump({
                'rep': output.clone().detach().cpu(),
                'layer': self.__layers_dict[layer]
            }, f)

    def load(self):
        """
        Loading all representations by the current datapoint key
        :return: dictionary matching layer's key to it's representation
        """
        representations_paths = glob.glob(os.path.join(self.__get_data_item_dir(), '*'))
        representations = {}

        for rep_path in representations_paths:
            with open(rep_path, 'rb') as f:
                data = pickle.load(f)
                rep = data['rep']
                layer_id = data['layer']
                representations[layer_id] = rep
            if self.__delete_after_load:
                os.remove(rep_path)
        os.removedirs(self.__get_data_item_dir())

        return representations

    def exists(self):
        return os.path.isdir(self.__get_data_item_dir())

    def __get_data_item_dir(self):
        return os.path.join(self.__root_dir, self.__filename)

    def __get_rep_path(self, layer):
        layer_name = str(layer)
        path = os.path.join(self.__get_data_item_dir(), f'{self.__layers_dict[layer]}_{layer_name}.pkl')
        os.makedirs(os.path.dirname(path),exist_ok=True)
        return path
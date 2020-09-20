import pickle
import glob
import os


class FileSystemHook(object):
    """
    A representation hook class, pickling the reps to the FS
    """
    def __init__(self, layers_dict, root_dir, filename):
        self.__layers_dict = layers_dict
        self.__root_dir = root_dir
        self.filename = filename

    def set_data_point_key(self, key):
        """
        Setting the datapoint's key (for saving the reps for a customized location, later making it easier to load the data.
        :param key: The key to the data point
        :return: None
        """
        self.filename = key

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
                'rep': output.clone().detach(),
                'layer': 'output'
            }, f)

        with open(input_path, 'wb') as f:
            pickle.dump({
                'rep': input[0].clone().detach(),
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
                'rep': output.clone().detach(),
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

        return representations

    def exists(self):
        return os.path.isdir(self.__get_data_item_dir())

    def __get_data_item_dir(self):
        return os.path.join(self.__root_dir, self.filename)

    def __get_rep_path(self, layer):
        path = os.path.join(self.__get_data_item_dir(), f'{self.__layers_dict[layer]}_{str(layer)}.pkl')
        os.makedirs(os.path.dirname(path),exist_ok=True)
        return path
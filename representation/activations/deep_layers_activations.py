from representation.acquisition.representation_save_hook import FileSystemHook
from representation.acquisition.representation_extraction import RepresentationExtractor

import os.path
from tqdm import tqdm


class DeepLayersActivations(object):
    def __init__(self, reps_cache_path, image_loader, get_layers_dict):
        self.__reps_cache_path = reps_cache_path
        self.__image_loader = image_loader
        self.__get_layers_dict = get_layers_dict

    def compare_pairs(self, model, dataset_dir, imgs_list, progress_label):
        re = RepresentationExtractor(model,
                                     self.__get_layers_dict(model),
                                     FileSystemHook(self.__get_layers_dict(model), self.__reps_cache_path, delete_after_load=False))

        for im_path in tqdm(imgs_list, desc=progress_label):
            try:
                full_im_path = os.path.join(dataset_dir, im_path)
                im = self.__image_loader.load_image(full_im_path)
                im_key = im_path
                re.save_layers_representation(im, im_key)

            except:
                print(f'Error on {full_im_path}')

        del re
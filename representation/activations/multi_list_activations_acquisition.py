
class MultiListAcquisition(object):
    def __init__(self, imgs_types_to_lists, imgs_types_to_dir, imgs_activitions_acquire):
        self.__imgs_types_to_lists = imgs_types_to_lists
        self.__imgs_types_to_dir = imgs_types_to_dir
        self.__imgs_activitions_acquire = imgs_activitions_acquire

    def compare_lists(self, model):
        for img_type in self.__imgs_types_to_lists:
            self.__imgs_activitions_acquire.compare_pairs(model,
                     self.__imgs_types_to_dir[img_type],
                     self.__imgs_types_to_lists[img_type],
                     img_type)


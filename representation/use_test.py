from representation.acquisition.model_layer_dicts.raw_model_layers_dict import get_model_layers_dict
from representation.acquisition.representation_extraction import RepresentationExtractor
from representation.acquisition.representation_save_hook import FileSystemHook
from representation.analysis.datapoints_rep_compare import DatapointsRepComparer
from representation.analysis.metrics.euclidian_distance_compare import EuclidianDistanceCompare
import torchvision
import os
import torchvision.transforms as transforms
from PIL import Image
import pickle


if __name__ == '__main__':
    test_vgg = torchvision.models.vgg16()
    image_path = ''
    re = RepresentationExtractor(test_vgg, get_model_layers_dict(test_vgg), FileSystemHook(get_model_layers_dict(test_vgg),os.path.join('.', 'reps')))
    im_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tt = transforms.Compose([
        transforms.Resize(im_size),  # 256
        transforms.CenterCrop(im_size),  # 224
        transforms.ToTensor(),
        normalize,
    ])

    # im1 = Image.open('../tests/raw_test_dataset/class_1/blonde.jpg')
    # if im1.mode != 'RGB':
    #     im1 = im1.convert('RGB')
    # im1t = tt(im1)
    # im1t = im1t.unsqueeze(0)
    #
    # reps = re.get_layers_representation(im1t, f'blonde')

    im1 = Image.open('../tests/raw_test_dataset/class_1/blonde.jpg')
    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')
    im1t = tt(im1)
    im1t = im1t.unsqueeze(0)

    re.save_layers_representation(im1t, f'blonde')

    im1 = Image.open('../tests/raw_test_dataset/class_1/black-hair.jpg')
    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')
    im1t = tt(im1)
    im1t = im1t.unsqueeze(0)

    re.save_layers_representation(im1t, f'black-hair')

    comp = DatapointsRepComparer(representation_extractor=re, comparison=EuclidianDistanceCompare())
    comparison = comp.compare_datapoints('blonde', 'black-hair')
    with open('./reps/comp.pkl', 'wb') as f:
        pickle.dump(comparison, f)
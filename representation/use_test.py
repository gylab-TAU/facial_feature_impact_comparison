from representation.raw_model_layers_dict import get_model_layers_dict
from representation.representation_extraction import RepresentationExtractor
from representation.representation_save_hook import FileSystemHook
import torchvision
import torch
import os
import torchvision.transforms as transforms
from PIL import Image, ImageOps


if __name__ == '__main__':
    test_vgg = torchvision.models.vgg16()
    image_path = ''
    re = RepresentationExtractor(test_vgg, get_model_layers_dict(test_vgg), FileSystemHook(get_model_layers_dict(test_vgg),os.path.join('.', 'reps'), 'blonde'))
    im_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tt = transforms.Compose([
        transforms.Resize(im_size),  # 256
        transforms.CenterCrop(im_size),  # 224
        transforms.ToTensor(),
        normalize,
    ])

    im1 = Image.open('../tests/raw_test_dataset/class_1/blonde.jpg')

    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')

    im1t = tt(im1)
    im1t = im1t.unsqueeze(0)

    for i in range(50):
        image = torch.randn(1, 3, 224, 224)
        reps = re.get_layers_representation(image, f'rand_image{i}')
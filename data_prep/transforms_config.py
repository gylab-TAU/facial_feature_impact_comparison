from torchvision import transforms
import json


def get_transforms(transform_config):
    if transform_config['transforms_type'] == 'standard':
        im_size = int(transform_config['image_size'])
        input_size = int(transform_config['net_input_size'])
        dataset_mean = json.loads(transform_config['dataset_mean'])
        dataset_std = json.loads(transform_config['dataset_std'])
        normalize = transforms.Normalize(dataset_mean, dataset_std),
        return {'train': transforms.Compose([
                        transforms.Resize(im_size),
                        transforms.CenterCrop(input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
            ]),
            'test': transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize])
        }
    elif transform_config['transforms_type'] == 'random_resize':
        im_size = int(transform_config['image_size'])
        input_size = int(transform_config['net_input_size'])
        dataset_mean = json.loads(transform_config['dataset_mean'])
        dataset_std = json.loads(transform_config['dataset_std'])
        normalize = transforms.Normalize(dataset_mean, dataset_std),
        return {'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
            'test': transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize])
        }

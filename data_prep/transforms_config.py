from torchvision import transforms
from const import TRAIN_PHASE, TEST_PHASE
import json


def get_transforms(transform_config):
    if transform_config['transforms_type'] == 'mtcnn':
        im_size = int(transform_config['image_size'])
        input_size = int(transform_config['net_input_size'])
        dataset_mean = json.loads(transform_config['dataset_means'])
        dataset_std = json.loads(transform_config['dataset_stds'])
        normalize = transforms.Normalize(dataset_mean, dataset_std)
        return {TRAIN_PHASE: transforms.Compose([
                        transforms.Resize([im_size, im_size]),
                        transforms.RandomCrop(input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize]),
            TEST_PHASE: transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize])
        }
    elif transform_config['transforms_type'] == 'dlib':
        im_size = int(transform_config['image_size'])
        content_size = int(transform_config['content_size'])
        input_size = int(transform_config['net_input_size'])
        dataset_mean = json.loads(transform_config['dataset_means'])
        dataset_std = json.loads(transform_config['dataset_stds'])
        normalize = transforms.Normalize(dataset_mean, dataset_std)
        return {TRAIN_PHASE: transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.CenterCrop(content_size),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
            TEST_PHASE: transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize])
        }
    elif transform_config['transforms_type'] == 'random_resize':
        im_size = int(transform_config['image_size'])
        input_size = int(transform_config['net_input_size'])
        dataset_mean = json.loads(transform_config['dataset_means'])
        dataset_std = json.loads(transform_config['dataset_stds'])
        normalize = transforms.Normalize(dataset_mean, dataset_std)
        return {TRAIN_PHASE: transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
            TEST_PHASE: transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize])
        }
    elif transform_config['transforms_type'] == 'birds':
        im_size = int(transform_config['image_size'])
        input_size = int(transform_config['net_input_size'])
        dataset_mean = json.loads(transform_config['dataset_means'])
        dataset_std = json.loads(transform_config['dataset_stds'])
        normalize = transforms.Normalize(dataset_mean, dataset_std)
        return {TRAIN_PHASE: transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
            TEST_PHASE: transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize])
        }

from torchvision import transforms
import torch
from const import TRAIN_PHASE, TEST_PHASE
import json
import mlflow


def mtcnn_transforms(transform_config):
    """
    Creating MTCNN image transformations (assuming images are MTCNN-aligned)
    Training:
    1. Resize to (image_size, image_size) pixels
    2. Random crop to (net_input_size, net_input_size) pixels (according to the network's input size)
    3. Applying random horizontal flip
    4. Transforming to tensor type
    5. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
    Validation:
    1. Resize to (image_size, image_size) pixels
    2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
    3. Transforming to tensor type
    4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
    """
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


def dlib_tranforms(transform_config):
    """
        Creating DLIB image transformations (assuming images are DLIB-aligned)
        Training:
        1. Resize to (image_size, image_size) pixels
        2. Center crop to (content_size, content_size) pixels
        3. Random crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        4. Applying random horizontal flip
        5. Transforming to tensor type
        6. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
        Validation:
        1. Resize to (image_size, image_size) pixels
        2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        3. Transforming to tensor type
        4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
        """
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


def random_resize_transforms(transform_config):
    """
    Creating random resize transforms (imagenet)
    Training:
        1. Resize to (image_size, image_size) pixels
        2. Random resized crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        with default ratio and scale
        3. Applying random horizontal flip
        4. Transforming to tensor type
        5. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
        Validation:
        1. Resize to (image_size, image_size) pixels
        2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        3. Transforming to tensor type
        4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
    """
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


def birds_transforms(transform_config):
    """
        Creating transforms for birds classification
        Training:
            1. Resize to (image_size, image_size) pixels
            2. Apply random rotation for +-30 degrees
            3. Random resized crop to (net_input_size, net_input_size) pixels (according to the network's input size)
            with default ratio and scale=(0.8,1.0)
            4. Applying random horizontal flip
            5. Transforming to tensor type
            6. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
            Validation:
            1. Resize to (image_size, image_size) pixels
            2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
            3. Transforming to tensor type
            4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
        """
    im_size = int(transform_config['image_size'])
    input_size = int(transform_config['net_input_size'])
    dataset_mean = json.loads(transform_config['dataset_means'])
    dataset_std = json.loads(transform_config['dataset_stds'])
    normalize = transforms.Normalize(dataset_mean, dataset_std)
    return {TRAIN_PHASE: transforms.Compose([
        transforms.Resize([im_size, im_size]),
        transforms.RandomRotation(40),
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



def birds_transforms2(transform_config):
    """
        Creating transforms for birds classification
        Training:
            1. Resize to (image_size, image_size) pixels
            2. Apply random rotation for +-30 degrees
            3. Random resized crop to (net_input_size, net_input_size) pixels (according to the network's input size)
            with default ratio and scale=(0.8,1.0)
            4. Applying random horizontal flip
            5. Transforming to tensor type
            6. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
            Validation:
            1. Resize to (image_size, image_size) pixels
            2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
            3. Transforming to tensor type
            4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
        """
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


def sociable_weavers_transforms(transform_config):
    """
    Creating transforms for birds classification
    Training:
        1. Resize to (image_size, image_size) pixels
        2. Apply random rotation for +-[rotation_angle] degrees
        3. Apply with [transforms_prob] gaussian blurring with [blur_kernel_size] and variance~U[[min_noise_variance],[max_noise_variance]]
        4. Random resized crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        with ratio=(1.0,1.0) and scale=(0.8,1.2)
        5. Applying random horizontal flip
        6. Transforming to tensor type
        7. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
        Validation:
        1. Resize to (image_size, image_size) pixels
        2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        3. Transforming to tensor type
        4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
    """
    im_size = int(transform_config['image_size'])
    input_size = int(transform_config['net_input_size'])
    dataset_mean = json.loads(transform_config['dataset_means'])
    dataset_std = json.loads(transform_config['dataset_stds'])
    normalize = transforms.Normalize(dataset_mean, dataset_std)

    return {
        TRAIN_PHASE: transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.RandomRotation(40),
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


def sociable_weavers_transforms2(transform_config):
    """
    Creating transforms for birds classification
    Training:
        1. Resize to (image_size, image_size) pixels
        2. Apply random rotation for +-[rotation_angle] degrees
        3. Apply with [transforms_prob] gaussian blurring with [blur_kernel_size] and variance~U[[min_noise_variance],[max_noise_variance]]
        4. Random resized crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        with ratio=(1.0,1.0) and scale=(0.8,1.2)
        5. Applying random horizontal flip
        6. Transforming to tensor type
        7. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
        Validation:
        1. Resize to (image_size, image_size) pixels
        2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
        3. Transforming to tensor type
        4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
    """
    mlflow.log_params({
        'rotation_angle': int(transform_config['rotation_angle']),
        'blur_kernel_size': int(transform_config['blur_kernel_size']),
    })
    im_size = int(transform_config['image_size'])
    input_size = int(transform_config['net_input_size'])
    dataset_mean = json.loads(transform_config['dataset_means'])
    dataset_std = json.loads(transform_config['dataset_stds'])
    normalize = transforms.Normalize(dataset_mean, dataset_std)
    transforms_prob = float(transform_config['transforms_prob'])
    min_std = float(transform_config['blur_min_std'])
    max_std = float(transform_config['blur_max_std'])
    mlflow.log_param('Gaussian blur kernel min std', min_std)
    mlflow.log_param('Gaussian blur kernel max std', max_std)
    max_noise_variance = float(transform_config['max_noise_variance'])
    min_noise_variance = float(transform_config['min_noise_variance'])

    return {
        TRAIN_PHASE: transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.RandomRotation(int(transform_config['rotation_angle'])),
            transforms.RandomApply(
                [transforms.GaussianBlur(int(transform_config['blur_kernel_size']), sigma=(min_std, max_std))],
                p=transforms_prob),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.RandomApply([transforms.Lambda(
            #     lambda x: x + (float(torch.rand(1)) * (
            #                 max_noise_variance - min_noise_variance) + min_noise_variance ** 0.5) * torch.randn(3,
            #                                                                                                     input_size,
            #                                                                                                     input_size))],
            #     p=transforms_prob),
            normalize]),
        TEST_PHASE: transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize])
    }


def whales_transforms(transform_config):
    """
       Creating transforms for whales classification
       Training:
           1. Resize to (image_size, image_size) pixels
           2. Apply random rotation for +-[rotation_angle] degrees
           3. Apply with [transforms_prob] gaussian blurring with [blur_kernel_size] and variance~U[[min_noise_variance],[max_noise_variance]]
           4. Random resized crop to (net_input_size, net_input_size) pixels (according to the network's input size)
           with ratio=(1.0,1.0) and scale=(0.8,1.2)
           5. Applying random horizontal flip
           6. Transforming to tensor type
           7. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
           Validation:
           1. Resize to (image_size, image_size) pixels
           2. Center crop to (net_input_size, net_input_size) pixels (according to the network's input size)
           3. Transforming to tensor type
           4. Normalizing according to mu=[dataset_means], sigma=[dataset_stds]
       """
    mlflow.log_params({
        'rotation_angle': int(transform_config['rotation_angle']),
        'blur_kernel_size': int(transform_config['blur_kernel_size']),
    })
    im_size = int(transform_config['image_size'])
    input_size = int(transform_config['net_input_size'])
    dataset_mean = json.loads(transform_config['dataset_means'])
    dataset_std = json.loads(transform_config['dataset_stds'])
    normalize = transforms.Normalize(dataset_mean, dataset_std)
    transforms_prob = float(transform_config['transforms_prob'])
    min_std = float(transform_config['blur_min_std'])
    max_std = float(transform_config['blur_max_std'])
    mlflow.log_param('Gaussian blur kernel min std', min_std)
    mlflow.log_param('Gaussian blur kernel max std', max_std)

    return {
        TRAIN_PHASE: transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.RandomRotation(int(transform_config['rotation_angle'])),
            # transforms.RandomApply(
                # [transforms.GaussianBlur(int(transform_config['blur_kernel_size']), sigma=(min_std, max_std))],
                # p=transforms_prob),
            transforms.RandomResizedCrop(input_size),#, scale=(0.5, 1.2)),#, ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomApply(
                [transforms.RandomPerspective()],
                p=transforms_prob),
            transforms.ToTensor(),
            transforms.RandomApply(
                [transforms.Grayscale(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))],
                p=0.3),
            normalize]),
        TEST_PHASE: transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize])
    }


def get_transforms(transform_config):
    """
    Generating image transforms according to [DATASET] parameters in configuration
    parameter [DATASET][transforms_type] says which transforms to use
    other parameters are used according to the specific [transforms_type]
    """
    if 'transforms_type' not in transform_config:
        transform_config['transforms_type'] = 'mtcnn'
        print('transforms_type does not appear in config. Assuming "mtcnn" transforms (standard)')
    if transform_config['transforms_type'] == 'mtcnn':
        return mtcnn_transforms(transform_config)
    elif transform_config['transforms_type'] == 'dlib':
        return dlib_tranforms(transform_config)
    elif transform_config['transforms_type'] == 'random_resize':
        return random_resize_transforms(transform_config)
    elif transform_config['transforms_type'] == 'birds':
        return birds_transforms(transform_config)
    elif transform_config['transforms_type'] == 'sociable_weavers':
        return sociable_weavers_transforms(transform_config)
    elif transform_config['transforms_type'] == 'whales':
        return whales_transforms(transform_config)

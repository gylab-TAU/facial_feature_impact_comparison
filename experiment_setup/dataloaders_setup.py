import torch.cuda

import const
from const import TRAIN_PHASE
from data_prep.transforms_config import get_transforms
import json
import os
import torch.utils.data as data
import data_prep.Datasets.finetuning_dataset


def dataloaders_setup(config, processed_dataset, image_loader, triplet=False):
    phase_size_dict = json.loads(config['DATASET']['phase_size_dict'])

    dataloaders = {}
    for phase in phase_size_dict:
        is_train = phase == TRAIN_PHASE
        ds_path = os.path.join(processed_dataset, phase)
        # image_folder = image_loader.load_dataset(ds_path, center_crop=not is_train)
        context = None
        if 'context' in config['DATASET']:
            context = config['DATASET']['context']
        image_folder = image_loader.load_dataset(ds_path, test=not is_train, triplet=triplet, context=context)
        if 'FINETUNING' in config:
            if config['FINETUNING']['classes_mode'] == 'append':
                image_folder = data_prep.Datasets.finetuning_dataset.FinetuningDataset(image_folder, int(config['MODELLING']['num_classes']))
        # print(len(image_folder))
        pin_memory = torch.cuda.is_available() and not const.DEBUG
        dataloaders[phase] = data.DataLoader(
            image_folder,
            batch_size=int(config['MODELLING']['batch_size']),
            num_workers=int(config['MODELLING']['workers']),
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=False)


    return dataloaders

from const import TRAIN_PHASE
import json
import os
import torch.utils.data as data
import data_prep.Datasets.finetuning_dataset


def dataloaders_setup(config, processed_dataset, image_loader):
    phase_size_dict = json.loads(config['DATASET']['phase_size_dict'])

    dataloaders = {}
    for phase in phase_size_dict:
        is_train = phase == TRAIN_PHASE
        ds_path = os.path.join(processed_dataset, phase)
        image_folder = image_loader.load_dataset(ds_path, center_crop=not is_train)
        if 'FINETUNING' in config:
            image_folder = data_prep.Datasets.finetuning_dataset.FinetuningDataset(image_folder, int(config['MODELLING']['num_classes']))
        print(len(image_folder))
        dataloaders[phase] = data.DataLoader(
            image_folder,
            batch_size=int(config['MODELLING']['batch_size']),
            num_workers=int(config['MODELLING']['workers']),
            shuffle=True,
            pin_memory=True,
            drop_last=False)

    return dataloaders

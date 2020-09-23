from const import TRAIN_PHASE
import json
import os
import torch.utils.data as data


def dataloaders_setup(config, processed_dataset, image_loader):
    phase_size_dict = json.loads(config['DATASET']['phase_size_dict'])

    dataloaders = {}
    for phase in phase_size_dict:
        is_train = phase != TRAIN_PHASE
        dataloaders[phase] = data.DataLoader(
            image_loader.load_dir(os.path.join(processed_dataset, phase), center_crop=is_train),
            batch_size=int(config['MODELLING']['batch_size']),
            num_workers=int(config['MODELLING']['workers']),
            shuffle=is_train,
            pin_memory=True,
            drop_last=is_train)

    return dataloaders

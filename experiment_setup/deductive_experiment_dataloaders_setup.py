from const import TRAIN_PHASE, VAL_PHASE
import os
import torch.utils.data as data


def dataloaders_setup(config, processed_dataset, image_loader, num_ids: int = 0, num_images: int = 0):
    if num_ids > 0:
        dataloaders = {}
        train_path = os.path.join(processed_dataset, TRAIN_PHASE)
        val_path = os.path.join(processed_dataset, VAL_PHASE)
        train_image_folder = image_loader.load_random_dataset(train_path, num_ids, num_images, center_crop=False)
        val_image_folder = image_loader.load_ids_dataset(val_path, train_image_folder.ids, center_crop=True)
        print(len(train_image_folder))
        dataloaders[TRAIN_PHASE] = data.DataLoader(
            train_image_folder,
            batch_size=int(config['MODELLING']['batch_size']),
            num_workers=int(config['MODELLING']['workers']),
            shuffle=True,
            pin_memory=True,
            drop_last=False)
        dataloaders[VAL_PHASE] = data.DataLoader(
            val_image_folder,
            batch_size=int(config['MODELLING']['batch_size']),
            num_workers=int(config['MODELLING']['workers']),
            shuffle=False,
            pin_memory=True,
            drop_last=False)
    else:
        dataloaders = dataloaders_setup.dataloaders_setup(config, processed_dataset, image_loader)

    return dataloaders

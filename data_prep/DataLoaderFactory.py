import torch.utils.data as data


class DataLoaderFactory(object):
    def __init__(self, batch_size, workers, shuffle=False, drop_last=False, center_crop=True, pin_memory=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.center_crop = center_crop
        self.workers = workers
        self.pin_memory = pin_memory

    def get_data_loader(self, data_location):
        return data.dataloader(
            self.image_loader.load_dir(data_location, self.center_crop),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.workers,
            pin_memory=self.pin_memory
        )
import torch
import os


class LocalModelStore(object):
    def __init__(self, arch, experiment_name, root_dir):
        self.__model_store_path = os.path.join(root_dir, experiment_name, arch, 'models')

    def save_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, acc: float, is_best: bool):
        path = self.__get_model_path(self.__get_model_filename(epoch, is_best))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'bw') as f:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict()}, f)
            if is_best:
                self.save_model(model, optimizer, epoch, acc, False)

    def load_model_and_optimizer(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, epoch: int = 0):
        path = self.__get_model_path(self.__get_model_filename(epoch, epoch == 0))
        return self.load_model_and_optimizer_loc(model, optimizer=optimizer, model_location=path)

    def load_model_and_optimizer_loc(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, model_location=None):
        with open(model_location, 'br') as f:
            model_checkpoint = torch.load(f)
            print(model.state_dict())
            model.load_state_dict(model_checkpoint['state_dict'])
            print(model.state_dict())
            if optimizer is not None:
                optimizer.load_state_dict(model_checkpoint['optimizer'])

        return model, optimizer, model_checkpoint['acc'], model_checkpoint['epoch']

    def __get_model_filename(self, epoch, is_best):
        if is_best:
            return 'best.pth'
        return f'{epoch}.pth'

    def __get_model_path(self, filename):
        return os.path.join(self.__model_store_path, filename)

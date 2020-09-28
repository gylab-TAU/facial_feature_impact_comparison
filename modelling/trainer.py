# from average_meter import AverageMeter
# from progress_meter import ProgressMeter
import time
import torch
from tqdm import tqdm
import const
# from const import TRAIN_PHASE, VAL_PHASE
# from metrics import accuracy
# from util import save_checkpoint


class Trainer(object):
    def __init__(self, model, criterion, optimizer, lr_scheduler, model_store, best_acc1:float = 0):
        self.model = model
        self.__criterion = criterion
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.__model_store = model_store
        self.__best_acc1 = best_acc1

    def train_model(self, start_epoch, end_epoch, data_loaders):
        for epoch in range(start_epoch, end_epoch):

            # modelling for one epoch
            self.__per_phase(epoch, const.TRAIN_PHASE, data_loaders)
            phase_loss, phase_acc = self.__per_phase(epoch, const.VAL_PHASE, data_loaders)

            self.__lr_scheduler.step()

            # remember best acc@1 and save checkpoint
            is_best = phase_acc > self.__best_acc1
            self.__best_acc1 = max(phase_acc, self.__best_acc1)

            self.__model_store.save_model(self.model, self.__optimizer, epoch, self.__best_acc1, is_best)


    def __per_phase(self, epoch, phase, data_loaders):
        # batch_time, losses, top1, top5, data_time, progress = get_epoch_meters(self.train_loader, epoch)

        if hasattr(data_loaders[phase].sampler, 'indices'):
            phase_size = len(data_loaders[phase].sampler.indices)
        else:
            phase_size = len(data_loaders[phase].dataset)

        phase_loss = 0
        phase_acc = 0

        # switch to modelling mode
        self.model.train(phase == const.TRAIN_PHASE)

        with torch.set_grad_enabled(phase == const.TRAIN_PHASE):
            for (images, target) in tqdm(data_loaders[phase], desc=phase):

                batch_loss, batch_acc = self.__per_batch(images, target)
                phase_loss += batch_loss / phase_size
                phase_acc += batch_acc.item() / phase_size

        return phase_loss, phase_acc

    def __per_batch(self, images, target):
        if torch.cuda.is_available():
            # TODO: Need to test this modification. before it used to check if GPU is not None. Why?
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = self.model(images)
        loss = self.__criterion(output, target)

        _, preds = torch.max(output, 1)
        # compute gradient and do optimizer step
        if self.model.training:
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

        return loss.data.item(), torch.sum(preds == target.data)
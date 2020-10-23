import torch
import const
import tqdm


class StandardEpochTrainer(object):
    def __init__(self, model: torch.nn.modules.Module,
                 dataloader: torch.utils.data.DataLoader,
                 phase: str,
                 num_batches_per_epoch_limit: int,
                 criterion, optimizer: torch.optim.Optimizer):
        self.model = model
        self.__optimizer = optimizer
        self.__dataloader = dataloader
        self.__phase = phase
        self.__num_batches_per_epoch_limit = num_batches_per_epoch_limit
        self.__criterion = criterion

    def run(self, epoch):
        phase_loss = 0
        phase_acc = 0
        num_batches = len(self.__dataloader)
        if 0 < self.__num_batches_per_epoch_limit < num_batches:
            num_batches = self.__num_batches_per_epoch_limit

        # switch to modelling mode
        self.model.train(self.__phase == const.TRAIN_PHASE)
        with torch.set_grad_enabled(self.__phase == const.TRAIN_PHASE):
            data_loader_iter = iter(self.__dataloader)
            for i in tqdm(range(num_batches), desc=self.__phase):
                (images, target) = next(data_loader_iter)

                batch_loss, batch_acc = self.__per_batch(images, target)
                if batch_loss >= 10:
                    # I once got a batchloss = Inf so I added this print to give a heads up
                    print('batch_loss >= 10', batch_loss)
                phase_loss += batch_loss / num_batches
                phase_acc += batch_acc / num_batches

        return phase_loss, phase_acc

    def __per_batch(self, images, target):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=False)
            target = target.cuda(non_blocking=False)

        # compute output
        output = self.model(images)
        loss = self.__criterion(output, target)

        _, preds = torch.max(output, 1)
        # compute gradient and do optimizer step
        if self.model.training:
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

        return loss.data.item(), torch.sum(preds == target.data).item() / target.shape[0]
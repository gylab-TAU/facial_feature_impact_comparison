import torch
import time
from tqdm import tqdm
import const
import pandas as pd
import os


class CustomTestTrainer(object):
    def __init__(self, model, criterion, optimizer, lr_scheduler, model_store, best_acc1:float = 0,
                 performance_tester=None, accuracy_threshold=0, num_epochs_to_test=None,
                 num_batches_per_epoch_limit=0, perf_logger=None, logs_path=None):
        self.model = model
        self.__criterion = criterion
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.__model_store = model_store
        self.__best_acc1 = best_acc1
        self.__performance_tester = performance_tester
        self.__performance_threshold = accuracy_threshold
        self.__num_epochs_to_test = num_epochs_to_test
        self.__num_batches_per_epoch_limit = num_batches_per_epoch_limit
        self.__performance_logger = perf_logger
        self.__logs_path = logs_path


        self.__avg_train_loss = []
        self.__train_acc = []
        self.__train_epochs = []
        self.__avg_val_loss = []
        self.__val_acc = []
        self.__val_epochs = []
        self.__lfw_acc = []
        self.__lfw_layer = []
        self.__lfw_thresh = []
        self.__lfw_epochs = []

    def train_model(self, start_epoch, end_epoch, data_loaders):
        for epoch in range(start_epoch, end_epoch):
            print(f'Epoch: {epoch}')

            # print("lr before train: ", self.__lr_scheduler.get_last_lr())
            # modelling for one epoch
            phase_loss, phase_acc = self.__per_phase(epoch, const.TRAIN_PHASE, data_loaders)
            print(f"average loss: {phase_loss}, epoch acc@1: {phase_acc}")
            self.__avg_train_loss.append(phase_loss)
            self.__train_acc.append(phase_acc)
            self.__train_epochs.append(epoch)
            self.__log_performance(epoch, const.TRAIN_PHASE)

            # print("lr after train: ", self.__lr_scheduler.get_last_lr())
            if type(self.__lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.__lr_scheduler.step(phase_loss)
            else:
                self.__lr_scheduler.step()
            # print("lr after train and step: ", self.__lr_scheduler.get_last_lr())

            # remember best acc@1 and save checkpoint
            is_best = phase_acc > self.__best_acc1
            self.__best_acc1 = max(phase_acc, self.__best_acc1)
            # if is_best:
            #     self.__model_store.save_model(self.model, self.__optimizer, epoch, self.__best_acc1, is_best)

            if self.__should_test(epoch+1):
                print("VAL")
                phase_loss, phase_acc = self.__per_phase(epoch, const.VAL_PHASE, data_loaders)
                self.__avg_val_loss.append(phase_loss)
                self.__val_acc.append(phase_acc)
                self.__val_epochs.append(epoch)
                self.__log_performance(epoch, const.VAL_PHASE)
                print(f"average loss: {phase_loss}, epoch acc@1: {phase_acc}")

                is_best = phase_acc > self.__best_acc1
                self.__best_acc1 = max(phase_acc, self.__best_acc1)
                save_start = time.perf_counter()
                self.__model_store.save_model(self.model, self.__optimizer, epoch, self.__best_acc1, is_best)
                print("save time: ", time.perf_counter()-save_start)
                bruto_test = time.perf_counter()
                perf = self.__test_performance(epoch+1)
                print("bruto test time: ", time.perf_counter()-bruto_test)
                if perf is not None:
                    print (f'Done in {epoch} epochs')
                    print(perf)
                    return perf

    def __log_performance(self, epoch, perf_type):
        if self.__logs_path is None:
            return
        os.makedirs(self.__logs_path, exist_ok=True)
        if perf_type == const.TRAIN_PHASE:
            pd.DataFrame({'epoch': self.__train_epochs, 'loss': self.__avg_train_loss,
                          'acc@1': self.__train_acc}).to_csv(os.path.join(self.__logs_path, 'train.csv'))
        elif perf_type == const.VAL_PHASE:
            pd.DataFrame({'epoch': self.__val_epochs, 'loss': self.__avg_val_loss,
                          'acc@1': self.__val_acc}).to_csv(os.path.join(self.__logs_path, 'val.csv'))
        elif perf_type == 'LFW':
            pd.DataFrame(
                {'epoch': self.__lfw_epochs, r'same\diff acc': self.__lfw_acc,
                 'layer': self.__lfw_layer, 'threshold': self.__lfw_thresh}).to_csv(os.path.join(self.__logs_path, 'lfw.csv'))


    def __test_performance(self, epoch):
        start_time = time.perf_counter()
        performance_df = self.__performance_tester.test_performance(self.model)
        print("neto test time: ", time.perf_counter() - start_time)
        self.__performance_logger.log_performance(epoch, performance_df)
        for layer in performance_df.index:
            accuracy = performance_df.loc[layer]['acc@1']
            threshold = performance_df.loc[layer]['threshold']
            self.__lfw_epochs.append(epoch)
            self.__lfw_acc.append(accuracy)
            self.__lfw_layer.append(layer)
            self.__lfw_thresh.append(threshold)
            self.__log_performance(epoch, 'LFW')
            print ("layer: ", layer, " accuracy: ", accuracy, " threshold: ", threshold)
            if accuracy > self.__performance_threshold:
                return layer, accuracy, threshold

        return None

    def __should_test(self, epoch):
        # If we have a way to test our performance
        # And if we defined a valid performance threshold
        # And if we have a valid "epoch step" after which we wish to test our performance
        # And if we the epoch is a natural number of 'epoch steps'
        # Dayeynu
        return self.__performance_tester is not None\
               and self.__performance_threshold != 0 \
               and self.__num_epochs_to_test is not None \
               and epoch % self.__num_epochs_to_test == 0

    def __per_phase(self, epoch, phase, data_loaders):
        # batch_time, losses, top1, top5, data_time, progress = get_epoch_meters(self.train_loader, epoch)
        phase_loss = 0
        phase_acc = 0
        num_batches = len(data_loaders[phase])
        if 0 < self.__num_batches_per_epoch_limit < num_batches:
            num_batches = self.__num_batches_per_epoch_limit

        # switch to modelling mode
        self.model.train(phase == const.TRAIN_PHASE)
        with torch.set_grad_enabled(phase == const.TRAIN_PHASE):
            data_loader_iter = iter(data_loaders[phase])
            pbar = tqdm(range(num_batches), desc=phase)
            for i in pbar:
                (images, target) = next(data_loader_iter)

                batch_loss, batch_acc = self.__per_batch(images, target)

                phase_loss += batch_loss / num_batches
                phase_acc += batch_acc / num_batches

                pbar.set_description(f"batch loss: { str(batch_loss)}, batch_acc: {str(batch_acc)}")

        return phase_loss, phase_acc

    def __per_batch(self, images, target):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = self.model(images) # DCNN
        if type(output) is not torch.Tensor:
            output = output[0]
        _, preds = torch.max(output, 1)
        batch_acc = torch.sum(preds == target.data).item() / target.shape[0]
        loss = self.__criterion(output, target)

        # compute gradient and do optimizer step
        if self.model.training:
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

        return loss.data.item(), batch_acc
from modelling.custom_test_trainer import CustomTestTrainer

from modelling.standard_test_trainer import StandardTestTrainer
from modelling.untested_trainer import UntestedTrainer


class TrainerFactory(object):
    def __init__(self, model_initializer, criterion_initializer, optimizer_initializer, lr_scheduler_initializer,
                 model_store):
        self.model_initializer = model_initializer
        self.criterion_initializer = criterion_initializer
        self.optimizer_initializer = optimizer_initializer
        self.lr_scheduler_initializer = lr_scheduler_initializer
        self.model_store = model_store

    def get_trainer(self, arch, optimizer_name, criterion_name, lr_scheduler_name, is_pretrained, num_classes,
                    checkpoint=None, epoch=0,
                    performance_tester=None,
                    performance_threshold=0,
                    num_epochs_to_test=None,
                    num_batches_per_epoch_limit=0, test_type: str = 'None'):
        model = self.model_initializer.get_model(arch, is_pretrained, num_classes)
        criterion = self.criterion_initializer.get_criterion(criterion_name)
        optimizer = self.optimizer_initializer.get_optimizer(optimizer_name, model)

        acc = 0
        if checkpoint is not None:
            model, optimizer, acc, epoch = self.model_store.load_model_and_optimizer_loc(model, optimizer, checkpoint)
        elif epoch != 0:
            model, optimizer, acc, epoch = self.model_store.load_model_and_optimizer(model, optimizer, epoch - 1)

        lr_scheduler = self.lr_scheduler_initializer.get_scheduler(lr_scheduler_name, optimizer, epoch)

        if test_type == 'LFW_TEST' and performance_tester != None:
            return CustomTestTrainer(model,
                                     criterion,
                                     optimizer,
                                     lr_scheduler,
                                     self.model_store,
                                     best_acc1=acc,
                                     performance_tester=performance_tester,
                                     accuracy_threshold=performance_threshold,
                                     num_epochs_to_test=num_epochs_to_test,
                                     num_batches_per_epoch_limit=num_batches_per_epoch_limit)

        elif test_type == 'standard':
            return StandardTestTrainer(model,
                                       criterion,
                                       optimizer,
                                       lr_scheduler,
                                       self.model_store,
                                       best_acc1=acc,
                                       accuracy_threshold=performance_threshold,
                                       num_epochs_to_test=num_epochs_to_test,
                                       num_batches_per_epoch_limit=num_batches_per_epoch_limit)
        else:
            return UntestedTrainer(model,
                                   criterion,
                                   optimizer,
                                   lr_scheduler,
                                   self.model_store,
                                   best_acc1=acc,
                                   num_batches_per_epoch_limit=num_batches_per_epoch_limit)

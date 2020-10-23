from modelling.performance_logger_stub import PerformanceLoggerStub
from modelling.multi_epoch_trainer import MultiEpochTrainer
from modelling.standard_epoch_trainer import StandardEpochTrainer
from modelling.custom_tester import CustomTester
import const

class CustomPhaseTrainerFactory(object):
    def __init__(self, model_initializer, criterion_initializer, optimizer_initializer, lr_scheduler_initializer,
                 model_store):
        self.model_initializer = model_initializer
        self.criterion_initializer = criterion_initializer
        self.optimizer_initializer = optimizer_initializer
        self.lr_scheduler_initializer = lr_scheduler_initializer
        self.model_store = model_store

    def get_trainer(self, arch, optimizer_name, criterion_name, lr_scheduler_name, is_pretrained, num_classes,
                    checkpoint=None, epoch=0,
                    performance_validator = None,
                    performance_tester=None,
                    performance_threshold=0,
                    num_epochs_to_test=None,
                    num_batches_per_epoch_limit=0, test_type: str = 'None', val_type:str = 'standard',
                    perf_logger=None, dataloaders:{} = None):
        model = self.model_initializer.get_model(arch, is_pretrained, num_classes)
        criterion = self.criterion_initializer.get_criterion(criterion_name)
        optimizer = self.optimizer_initializer.get_optimizer(optimizer_name, model)

        acc = 0
        if checkpoint is not None:
            model, optimizer, acc, epoch = self.model_store.load_model_and_optimizer_loc(model, optimizer,
                                                                                         checkpoint)
        elif epoch != 0:
            model, optimizer, acc, epoch = self.model_store.load_model_and_optimizer(model, optimizer, epoch - 1)

        lr_scheduler = self.lr_scheduler_initializer.get_scheduler(lr_scheduler_name, optimizer, epoch)

        if perf_logger is None:
            perf_logger = PerformanceLoggerStub()

        train_phase = StandardEpochTrainer(model,
                                           dataloaders[const.TRAIN_PHASE],
                                           const.TRAIN_PHASE,
                                           num_batches_per_epoch_limit,
                                           criterion,
                                           optimizer)

        if val_type == 'LFW_TEST' and performance_validator is not None:
            val_phase = CustomTester(model, performance_validator, perf_logger)
        elif val_type == 'standard' and performance_validator is not None:
            val_phase = StandardEpochTrainer(model, dataloaders[const.VAL_PHASE], const.VAL_PHASE,
                                             num_batches_per_epoch_limit, criterion, optimizer)

        if test_type == 'LFW_TEST' and performance_tester is not None:
            test_phase = CustomTester(model, performance_tester, perf_logger)
        elif test_type == 'standard' and performance_tester is not None:
            test_phase = StandardEpochTrainer(model, dataloaders[const.TEST_PHASE], const.TEST_PHASE,
                                             num_batches_per_epoch_limit, criterion, optimizer)

        return MultiEpochTrainer(model,
                                 optimizer,
                                 self.model_store,
                                 train_phase,
                                 val_phase,
                                 test_phase,
                                 lr_scheduler,
                                 performance_threshold,
                                 acc,
                                 num_epochs_to_test)

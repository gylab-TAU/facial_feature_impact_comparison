import mlflow
from modelling.custom_test_trainer import CustomTestTrainer
from modelling.performance_logger_stub import PerformanceLoggerStub
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

    def get_trainer(self,
                    arch: str,
                    optimizer_name: str, optimizer_params: {},
                    criterion_name: str, criterion_params: {},
                    lr_scheduler_name: str, lr_scheduler_params: {},
                    num_classes: int,
                    is_pretrained: bool = False,
                    epoch: int = 0,
                    checkpoint: str = None,
                    performance_tester=None,
                    performance_threshold: float = 1,
                    num_epochs_to_test: int = None,
                    num_batches_per_epoch_limit: int = 0, test_type: str = 'None',
                    perf_logger=None, logs_path=None, finetuning: bool = False):
        mlflow.log_param('optimizer', optimizer_name)
        mlflow.log_params(optimizer_params)
        mlflow.log_param('criterion', criterion_name)
        mlflow.log_params(criterion_params)
        mlflow.log_param('lr_scheduler', lr_scheduler_name)
        mlflow.log_params(lr_scheduler_params)
        mlflow.log_param('is_pretrained', is_pretrained)

        use_arcface = (criterion_name.lower() == 'arcface')
        model = self.model_initializer.get_model(arch, is_pretrained, num_classes, use_arcface)
        criterion = self.criterion_initializer.get_criterion(criterion_name, criterion_params)
        optimizer = self.optimizer_initializer.get_optimizer(optimizer_name, model, optimizer_params)

        acc = 0
        print(checkpoint)
        if checkpoint != None:
            model, _, acc, epoch = self.model_store.load_model_and_optimizer_loc(model, None, checkpoint)
            if finetuning:
                model, optimizer, acc, epoch = self.model_store.load_model_and_optimizer_loc(model, None, checkpoint)
                optimizer = self.optimizer_initializer.get_optimizer(optimizer_name, model, optimizer_params)
                epoch = 0
        elif epoch != 0:
            model, optimizer, acc, epoch = self.model_store.load_model_and_optimizer(model, optimizer, epoch - 1)

        lr_scheduler = self.lr_scheduler_initializer.get_scheduler(lr_scheduler_name, optimizer, lr_scheduler_params, epoch - 1)

        if perf_logger is None:
            perf_logger = PerformanceLoggerStub()

        # if arch == 'context_vgg16':
        #     return CustomTestTrainer(model,
        #                              criterion,
        #                              optimizer,
        #                              lr_scheduler,
        #                              self.model_store,
        #                              best_acc1=acc,
        #                              performance_tester=performance_tester,
        #                              accuracy_threshold=performance_threshold,
        #                              num_epochs_to_test=num_epochs_to_test,
        #                              num_batches_per_epoch_limit=num_batches_per_epoch_limit,
        #                              perf_logger=perf_logger, logs_path=logs_path)

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
                                     num_batches_per_epoch_limit=num_batches_per_epoch_limit,
                                     perf_logger=perf_logger, logs_path=logs_path)
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

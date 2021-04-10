import json
from modelling.factories.trainer_factory import TrainerFactory
from modelling.factories.model_initializer import ModelInitializer
from modelling.factories.criterion_initializer import CrossEntropyCriterionInitializer
from modelling.factories.optimizer_initializer import SGDOptimizerInitializer
from modelling.factories.lr_scheduler_initializer import LRSchedulerInitializer
from modelling.local_model_store import LocalModelStore
from modelling.performance_logger import PerformanceLogger
from modelling.performance_logger_stub import PerformanceLoggerStub


def get_trainer(config, num_classes, start_epoch, perf_tester=None):
    model_store = LocalModelStore(config['MODELLING']['architecture'],
                                  config['GENERAL']['experiment_name'],
                                  config['GENERAL']['root_dir'])
    try:
        checkpoint_path = config['MODELLING']['checkpoint_path']
        if checkpoint_path == '':
            checkpoint_path = None
    except:
        checkpoint_path = None
    trainer_factory = TrainerFactory(
        ModelInitializer(json.loads(config['MODELLING']['feature_parallelized_architectures'])),
        CrossEntropyCriterionInitializer(),
        SGDOptimizerInitializer(float(config['OPTIMIZING']['lr']),
                                float(config['OPTIMIZING']['momentum']),
                                float(config['OPTIMIZING']['weight_decay'])),
        LRSchedulerInitializer(int(config['OPTIMIZING']['step']),
                               float(config['OPTIMIZING']['gamma'])),
        model_store)

    perf_threshold = None
    num_epochs_to_test = None
    num_batches_per_epoch_limit = int(config['MODELLING']['num_batches_per_epoch_limit'])

    if perf_tester is not None:
        perf_threshold = float(config['MODELLING']['perf_threshold'])
        num_epochs_to_test = int(config['MODELLING']['num_epochs_to_test'])

    perf_logger = get_performance_logger(config)

    trainer = trainer_factory.get_trainer(config['MODELLING']['architecture'],
                                          config['OPTIMIZING']['optimizer'],
                                          config['MODELLING']['criterion_name'],
                                          config['OPTIMIZING']['lr_scheduler'],
                                          config['MODELLING']['is_pretrained'] == 'True',
                                          num_classes, epoch=start_epoch,
                                          checkpoint=checkpoint_path,
                                          performance_tester=perf_tester,
                                          performance_threshold=perf_threshold,
                                          num_epochs_to_test=num_epochs_to_test,
                                          num_batches_per_epoch_limit=num_batches_per_epoch_limit,
                                          test_type=config['MODELLING']['performance_test'],
                                          perf_logger=perf_logger)

    return trainer


def get_performance_logger(config):
    if 'perf_log_path' in config['MODELLING']:
        return PerformanceLogger(config['MODELLING']['perf_log_path'])
    else:
        return PerformanceLoggerStub()

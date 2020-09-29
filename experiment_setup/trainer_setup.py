import json
from modelling.factories.trainer_factory import TrainerFactory
from modelling.factories.model_initializer import ModelInitializer
from modelling.factories.criterion_initializer import CrossEntropyCriterionInitializer
from modelling.factories.optimizer_initializer import SGDOptimizerInitializer
from modelling.factories.lr_scheduler_initializer import LRSchedulerInitializer
from modelling.local_model_store import LocalModelStore


def get_trainer(config, num_classes, start_epoch, perf_tester=None):
    model_store = LocalModelStore(config['MODELLING']['architecture'],
                                  config['GENERAL']['experiment_name'],
                                  config['GENERAL']['root_dir'])

    trainer_factory = TrainerFactory(
        ModelInitializer(json.loads(config['MODELLING']['feature_parallelized_architectures'])),
        CrossEntropyCriterionInitializer(),
        SGDOptimizerInitializer(float(config['OPTIMIZING']['lr']),
                                float(config['OPTIMIZING']['momentum']),
                                float(config['OPTIMIZING']['weight_decay'])),
        LRSchedulerInitializer(int(config['OPTIMIZING']['step']),
                               float(config['OPTIMIZING']['gamma'])),
        model_store)

    if 'performance_test' in config['MODELLING']:
        perf_threshold = float(config['MODELLING']['perf_threshold'])
        num_epochs_to_test = int(config['MODELLING']['num_epochs_to_test'])
        num_batches_per_epoch_limit = int(config['MODELLING']['num_batches_per_epoch_limit'])

    trainer = trainer_factory.get_trainer(config['MODELLING']['architecture'],
                                          config['OPTIMIZING']['optimizer'],
                                          config['MODELLING']['criterion_name'],
                                          config['OPTIMIZING']['lr_scheduler'],
                                          bool(config['MODELLING']['is_pretrained']),
                                          num_classes, epoch=start_epoch,
                                          performance_tester=perf_tester,
                                          performance_threshold=perf_threshold,
                                          num_epochs_to_test=num_epochs_to_test,
                                          num_batches_per_epoch_limit=num_batches_per_epoch_limit)

    return trainer
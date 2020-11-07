import json
from modelling.factories.parameterized_trainer_factory import TrainerFactory
from modelling.factories.model_initializer import ModelInitializer
from modelling.factories.reflection.generic_criterion_initializer import GenericCriterionInitializer
from modelling.factories.reflection.generic_optimizer_initializer import GenericOptimizerInitializer
from modelling.factories.reflection.generic_lr_scheduler_initializer import GenericLRSchedulerInitializer
from modelling.local_model_store import LocalModelStore


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
        GenericCriterionInitializer(),
        GenericOptimizerInitializer(),
        GenericLRSchedulerInitializer(),
        model_store)

    perf_threshold = None
    num_epochs_to_test = None
    num_batches_per_epoch_limit = int(config['MODELLING']['num_batches_per_epoch_limit'])

    if perf_tester is not None:
        perf_threshold = float(config['MODELLING']['perf_threshold'])
        num_epochs_to_test = int(config['MODELLING']['num_epochs_to_test'])

    perf_test_name = 'None'
    try:
        perf_test_name = config['MODELLING']['performance_test']
    except:
        perf_test_name = 'None'

    trainer = trainer_factory.get_trainer(config['MODELLING']['architecture'],
                                          config['OPTIMIZING']['optimizer'], json.loads(config['OPTIMIZING']['optimizer_params']),
                                          config['MODELLING']['criterion_name'], json.loads(config['MODELLING']['criterion_params']),
                                          config['OPTIMIZING']['lr_scheduler'], json.loads(config['MODELLING']['lr_scheduler_params']),
                                          num_classes,
                                          config['MODELLING']['is_pretrained'] == 'True',
                                          epoch=start_epoch,
                                          checkpoint=checkpoint_path,
                                          performance_tester=perf_tester,
                                          performance_threshold=perf_threshold,
                                          num_epochs_to_test=num_epochs_to_test,
                                          num_batches_per_epoch_limit=num_batches_per_epoch_limit, test_type=perf_test_name)

    return trainer

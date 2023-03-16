import json
import mlflow
from modelling.factories.parameterized_trainer_factory import TrainerFactory
from modelling.factories.reflection.custom_criterion_initializer import CustomCriterionInitializer
from modelling.factories.reflection.generic_optimizer_initializer import GenericOptimizerInitializer
from modelling.factories.reflection.generic_lr_scheduler_initializer import GenericLRSchedulerInitializer
from modelling.local_model_store import LocalModelStore
from modelling.performance_logger import PerformanceLogger
from modelling.performance_logger_stub import PerformanceLoggerStub
from modelling.factories.arcface.arcface_model_initializier import ArcFaceModelInitializer
from modelling.factories.verification.verification_model_initializer import VerificationModelInitializer
from representation.acquisition.model_layer_dicts.reflection_factory import ReflectionFactory


def get_trainer(config, num_classes, start_epoch, perf_tester=None):
    model_store = LocalModelStore(config['MODELLING']['architecture'],
                                  config['GENERAL']['experiment_name'],
                                  config['GENERAL']['root_dir'])
    checkpoint_path_param_name = 'checkpoint_path'
    checkpoint_path = None
    if checkpoint_path_param_name in config['MODELLING']:
        checkpoint_path = config['MODELLING'][checkpoint_path_param_name]
        if checkpoint_path == '':
            checkpoint_path = None
        else:
            mlflow.log_param('checkpoint_path', checkpoint_path)
    loss_factory = CustomCriterionInitializer()
    trainer_factory = TrainerFactory(
        VerificationModelInitializer(
            ArcFaceModelInitializer(json.loads(config['MODELLING']['feature_parallelized_architectures'])),
            ReflectionFactory().get_dict_extractor(config['MODELLING']['rep_layers']),
            config['MODELLING']['reps_cache_path'],
            loss_factory.get_criterion(config['MODELLING']['verification_score_name'])),
        CustomCriterionInitializer(),
        GenericOptimizerInitializer(),
        GenericLRSchedulerInitializer(),
        model_store)

    perf_threshold = None
    num_epochs_to_test = None
    num_batches_per_epoch_limit = int(config['MODELLING']['num_batches_per_epoch_limit'])
    mlflow.log_param('num_batches_per_epoch_limit', num_batches_per_epoch_limit)

    if perf_tester is not None:
        perf_threshold = float(config['MODELLING']['perf_threshold'])
        mlflow.log_param('perf_threshold', perf_threshold)
        num_epochs_to_test = int(config['MODELLING']['num_epochs_to_test'])
        mlflow.log_param('num_epochs_to_test', num_epochs_to_test)

    perf_test_param_name = 'performance_test'
    perf_test_name = 'None'
    if perf_test_param_name in config['MODELLING']:
        perf_test_name = config['MODELLING'][perf_test_param_name]

    perf_logger = get_performance_logger(config)

    logs_path = None
    if 'logs_path' in config['MODELLING']:
        logs_path = config['MODELLING']['logs_path']

    trainer = trainer_factory.get_trainer(config['MODELLING']['architecture'],
                                          config['OPTIMIZING']['optimizer'], json.loads(config['OPTIMIZING']['optimizer_params']),
                                          config['MODELLING']['criterion_name'], json.loads(config['MODELLING']['criterion_params']),
                                          config['OPTIMIZING']['lr_scheduler'], json.loads(config['OPTIMIZING']['lr_scheduler_params']),
                                          num_classes,
                                          config['MODELLING']['is_pretrained'] == 'True',
                                          epoch=start_epoch,
                                          checkpoint=checkpoint_path,
                                          performance_tester=perf_tester,
                                          performance_threshold=perf_threshold,
                                          num_epochs_to_test=num_epochs_to_test,
                                          num_batches_per_epoch_limit=num_batches_per_epoch_limit, test_type=perf_test_name,
                                          perf_logger=perf_logger, logs_path=logs_path, finetuning=('FINETUNING' in config))

    return trainer


def get_performance_logger(config):
    if 'perf_log_path' in config['MODELLING']:
        return PerformanceLogger(config['MODELLING']['perf_log_path'])
    else:
        return PerformanceLoggerStub()
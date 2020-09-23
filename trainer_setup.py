import json
from modelling.trainer_factory import TrainerFactory
from modelling.model_initializer import ModelInitializer
from modelling.criterion_initializer import CrossEntropyCriterionInitializer
from modelling.optimizer_initializer import SGDOptimizerInitializer
from modelling.lr_scheduler_initializer import LRSchedulerInitializer
from modelling.local_model_store import LocalModelStore


def get_trainer(config, num_classes, start_epoch):
    model_store = LocalModelStore(config['MODELLING']['architecture'],
                                  config['GENERAL']['root_dir'],
                                  config['GENERAL']['experiment_name'])



    trainer_factory = TrainerFactory(
        ModelInitializer(json.loads(config['MODELLING']['feature_parallelized_architectures'])),
        CrossEntropyCriterionInitializer(),
        SGDOptimizerInitializer(float(config['OPTIMIZING']['lr']),
                                float(config['OPTIMIZING']['momentum']),
                                float(config['OPTIMIZING']['weight_decay'])),
        LRSchedulerInitializer(int(config['OPTIMIZING']['step']),
                               float(config['OPTIMIZING']['gamma'])),
        model_store)

    trainer = trainer_factory.get_trainer(config['MODELLING']['architecture'],
                                          config['OPTIMIZING']['optimizer'],
                                          config['MODELLING']['criterion_name'],
                                          config['OPTIMIZING']['lr_scheduler'],
                                          bool(config['MODELLING']['is_pretrained']),
                                          num_classes, epoch=start_epoch)

    return trainer
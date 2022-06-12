from const import CONFIG_PATH
import mlflow
import const
import glob
import argparse
import time
import datetime
import os
import configparser
import json
# from data_prep.image_loader import ImageLoader
from data_prep.var_transform_image_loader import ImageLoader
from data_prep.transforms_config import get_transforms
from modelling.local_model_store import LocalModelStore
# from experiment_setup.lfw_test_setup import get_lfw_test
from experiment_setup.efficient_lfw_setup import get_lfw_test
from experiment_setup.dataset_filters_setup import setup_dataset_filter
from experiment_setup.dataloaders_setup import dataloaders_setup
from experiment_setup.triplet_trainer_setup import get_trainer
from experiment_setup.pairs_behaviour_setup import setup_pairs_reps_behaviour
import modelling.finetuning
import pandas as pd
from representation.analysis.rep_dist_mat import DistMatrixComparer

mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default=CONFIG_PATH, help='A path to the actual config file to use for the run')
    parser.add_argument("--config_dir", type=str, default=None, help='A path to the dir containing the configuration files (searches for files ending with .cfg)')
    parser.add_argument("--debug", action='store_true', help='when using --debug, stops from using the GPU (from most actions) in order to make debugging easier')

    args = parser.parse_args()
    return args


def run_experiment(config_path):
    # Get the configuration file
    print(datetime.datetime.now())
    start = time.perf_counter()
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('Running experiment {config_path}')
    config.read(config_path)
    print()
    print("Running experiment " + config['GENERAL']['experiment_name'])
    if mlflow.get_experiment_by_name(config['GENERAL']['experiment_name']) is None:
        mlflow.create_experiment(config['GENERAL']['experiment_name'], artifact_location=os.path.join(const.MLFLOW_ARTIFACT_STORE, config['GENERAL']['experiment_name']))
    mlflow.set_experiment(config['GENERAL']['experiment_name'])
    run_name = None
    if 'run_name' in config['GENERAL']:
        run_name = config['GENERAL']['run_name']

    with mlflow.start_run(run_name=run_name):
        print(mlflow.get_artifact_uri())
        mlflow.log_artifact(config_path)
        # Create image loader (by the configuration)
        im_size = json.loads(config['DATASET']['image_size'])
        mlflow.log_param('image_size', im_size)
        if 'post_crop_im_size' in config['DATASET']:
            post_crop_im_size = int(config['DATASET']['post_crop_im_size'])
        elif 'net_input_size' in config['DATASET']:
            post_crop_im_size = int(config['DATASET']['net_input_size'])
        mlflow.log_param('post_crop_im_size', post_crop_im_size)
        dataset_means = json.loads(config['DATASET']['dataset_means'])
        mlflow.log_param('dataset_means', dataset_means)
        dataset_stds = json.loads(config['DATASET']['dataset_stds'])
        mlflow.log_param('dataset_stds', dataset_stds)
        crop_scale = None
        if 'crop_scale' in config['DATASET']:
            crop_scale = json.loads(config['DATASET']['crop_scale'])
            crop_scale = (crop_scale['max'], crop_scale['min'])

        image_loader = ImageLoader(get_transforms(config['DATASET']))

        # Create the dataset filters by config (if they are needed)
        filter = setup_dataset_filter(config)

        # Activate the filters
        processed_dataset, num_classes = filter.process_dataset(
            config['DATASET']['raw_dataset_path'],
            config['DATASET']['dataset_name'])

        print("training on dataset: ", processed_dataset)
        mlflow.log_param('training_dataset', processed_dataset)
        # Create dataloader for the training
        triplet = config['MODELLING']['criterion_name'].lower() == 'triplet'
        dataloaders = dataloaders_setup(config, processed_dataset, image_loader, triplet=triplet)

        # Get access to pre trained models
        model_store = LocalModelStore(config['MODELLING']['architecture'],
                                      config['GENERAL']['root_dir'],
                                      config['GENERAL']['experiment_name'])
        mlflow.log_param('architecture', config['MODELLING']['architecture'])
        mlflow.log_param('experiment_name', config['GENERAL']['experiment_name'])
        mlflow.log_param('root_dir', config['GENERAL']['root_dir'])

        start_epoch = int(config['MODELLING']['start_epoch'])
        end_epoch = int(config['MODELLING']['end_epoch'])
        mlflow.log_param('start_epoch', start_epoch)
        mlflow.log_param('end_epoch', end_epoch)

        # creating the lfw tester
        lfw_tester = get_lfw_test(config, image_loader)

        num_classes = int(config['MODELLING']['num_classes'])
        mlflow.log_param('num_classes', num_classes)

        # Creating the trainer and loading the pretrained model if specified in the configuration
        trainer = get_trainer(config, num_classes, start_epoch, lfw_tester)

        # If we wish to finetune a pretrained model on the new dataset
        if 'FINETUNING' in config:
            model = trainer.model
            print(config['FINETUNING']['classes_mode'])
            num_cls = int(config['FINETUNING']['num_classes'])
            mlflow.log_param('finetuning', True)
            if config['FINETUNING']['classes_mode'] == 'append':
                model = modelling.finetuning.append_classes(trainer.model, num_cls)
            elif config['FINETUNING']['classes_mode'] == 'replace':
                model = modelling.finetuning.replace_classes(trainer.model, num_cls)
            elif config['FINETUNING']['classes_mode'] == 'overlay':
                model = modelling.finetuning.overlay_classes(model, num_classes, num_cls)
            mlflow.log_param('finetuning_classes', num_cls)
            model = modelling.finetuning.freeze_layers(model, int(config['FINETUNING']['freeze_end']))
            mlflow.log_param('finetuning_classes_mode', config['FINETUNING']['classes_mode'])
            mlflow.log_param('freeze_depth', int(config['FINETUNING']['freeze_end']))
            trainer.model = model

        # Will train the model from start_epoch to (end_epoch - 1) with the given dataloaders
        trainer.train_model(start_epoch, end_epoch, dataloaders)

        flag = True
        if flag:
            if lfw_tester is not None:
                lfw_results = lfw_tester.test_performance(trainer.model)
                print(lfw_results)
                lfw_path = os.path.join(config['LFW_TEST']['reps_results_path'], config['LFW_TEST']['output_filename'])
                os.makedirs(config['LFW_TEST']['reps_results_path'], exist_ok=True)
                lfw_results.to_csv(lfw_path)
                mlflow.log_artifact(lfw_path)

        reps_behaviour_extractor = setup_pairs_reps_behaviour(config, image_loader)

        if reps_behaviour_extractor is not None:
            output = reps_behaviour_extractor.compare_lists(trainer.model)

            if output is not None:
                if type(output) == pd.DataFrame:
                    results_path = os.path.join(config['REP_BEHAVIOUR']['reps_results_path'],
                                                config['REP_BEHAVIOUR']['output_filename'] + '.csv')

                    print('Saving results in ', results_path)
                    os.makedirs(os.path.dirname(results_path), exist_ok=True)
                    output.to_csv(results_path)
                    mlflow.log_artifact(results_path)

        end = time.perf_counter()
        print(datetime.datetime.now())
        print((end - start) / 3600)
        print((time.process_time()) / 3600)
        print('done')

    # TODO: Divide the analysis from training and from the data prep - use dir tree as indicator to the model training


if __name__ == '__main__':
    args = get_args()
    const.DEBUG = args.debug
    if args.config_dir is not None:
        config_paths = glob.glob(os.path.join(args.config_dir, '*'))
        for path in config_paths:
            run_experiment(path)
    else:
        run_experiment(args.config_path)

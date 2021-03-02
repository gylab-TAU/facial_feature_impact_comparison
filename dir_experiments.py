from const import CONFIG_PATH
import glob
import argparse
import time
import datetime
import os
import configparser
import json
from data_prep.image_loader import ImageLoader
from modelling.local_model_store import LocalModelStore
from experiment_setup.lfw_test_setup import get_lfw_test
from experiment_setup.dataset_filters_setup import setup_dataset_filter
from experiment_setup.dataloaders_setup import dataloaders_setup
from experiment_setup.generic_trainer_setup import get_trainer
from experiment_setup.pairs_behaviour_setup import setup_pairs_reps_behaviour
from representation.analysis.rep_dist_mat import DistMatrixComparer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default=CONFIG_PATH)
    parser.add_argument("--config_dir", type=str, default=None)

    args = parser.parse_args()
    return args

def run_experiment(config_path):
    # Get the configuration file
    print(datetime.datetime.now())
    start = time.perf_counter()
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path)
    print("Running experiment " + config['GENERAL']['experiment_name'])

    # Create image loader (by the configuration)
    im_size = int(config['DATASET']['image_size'])
    post_crop_im_size = int(config['DATASET']['post_crop_im_size'])
    dataset_means = json.loads(config['DATASET']['dataset_means'])
    dataset_stds = json.loads(config['DATASET']['dataset_stds'])
    crop_scale=None
    if 'crop_scale' in config['DATASET']:
        crop_scale = json.loads(config['DATASET']['crop_scale'])
        crop_scale = (crop_scale['max'], crop_scale['min'])
    image_loader = ImageLoader(im_size, post_crop_im_size, dataset_means, dataset_stds, crop_scale=crop_scale)

    # Create the dataset filters by config (if they are needed)
    filter = setup_dataset_filter(config)

    # Activate the filters
    processed_dataset, num_classes = filter.process_dataset(
        config['DATASET']['raw_dataset_path'],
        config['DATASET']['dataset_name'])

    print("training on dataset: ", processed_dataset)

    # Super hardcoded to ignore specific bug. feel free to remove
    # num_classes = 500

    # Create dataloader for the training
    dataloaders = dataloaders_setup(config, processed_dataset, image_loader)

    # Get access to pre trained models
    model_store = LocalModelStore(config['MODELLING']['architecture'],
                                  config['GENERAL']['root_dir'],
                                  config['GENERAL']['experiment_name'])

    start_epoch = int(config['MODELLING']['start_epoch'])
    end_epoch = int(config['MODELLING']['end_epoch'])

    # creating the lfw tester
    lfw_tester = get_lfw_test(config, image_loader)

    # Creating the trainer and loading the pretrained model if specified in the configuration
    trainer = get_trainer(config, num_classes, start_epoch, lfw_tester)

    # Will train the model from start_epoch to (end_epoch - 1) with the given dataloaders
    trainer.train_model(start_epoch, end_epoch, dataloaders)

    lfw_results = lfw_tester.test_performance(trainer.model)
    print(lfw_results)


    reps_behaviour_extractor = setup_pairs_reps_behaviour(config, image_loader)
    if reps_behaviour_extractor != None:

        output = reps_behaviour_extractor.compare_lists(trainer.model)


        results_path = os.path.join(config['REP_BEHAVIOUR']['reps_results_path'],
                                    config['REP_BEHAVIOUR']['output_filename'] + '.csv')
        lfw_path = os.path.join(config['REP_BEHAVIOUR']['reps_results_path'], 'logs.csv')
        print('Saving results in ', results_path)
        os.makedirs(config['REP_BEHAVIOUR']['reps_results_path'], exist_ok=True)
        if output is not None:
            output.to_csv(results_path)
        lfw_results.to_csv(lfw_path)


    end = time.perf_counter()
    print(datetime.datetime.now())
    print((end - start) / 3600)
    print((time.process_time()) / 3600)
    print('done')

    # TODO: Add option to start from existing models.
    # TODO: Divide the analysis from training and from the data prep - use dir tree as indicator to the model training


if __name__ == '__main__':
    args = get_args()
    if args.config_dir is not None:
        config_paths = glob.glob(os.path.join(args.config_dir, '*.cfg'))
        for path in config_paths:
            run_experiment(path)
    else:
        run_experiment(args.config_path)

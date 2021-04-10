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
from modelling.factories.model_initializer import ModelInitializer
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
    results_root = config['REP_BEHAVIOUR']['reps_results_path']
    im_size = int(config['DATASET']['image_size'])
    post_crop_im_size = int(config['DATASET']['post_crop_im_size'])
    dataset_means = json.loads(config['DATASET']['dataset_means'])
    dataset_stds = json.loads(config['DATASET']['dataset_stds'])
    crop_scale=None
    if 'crop_scale' in config['DATASET']:
        crop_scale = json.loads(config['DATASET']['crop_scale'])
        crop_scale = (crop_scale['max'], crop_scale['min'])
    image_loader = ImageLoader(im_size, post_crop_im_size, dataset_means, dataset_stds, crop_scale=crop_scale)

    # Get access to pre trained models
    model_store = LocalModelStore(config['MODELLING']['architecture'],
                                  config['GENERAL']['root_dir'],
                                  config['GENERAL']['experiment_name'])

    path_to_models = config['GENERAL']['weights_dir']
    ids_dirs = glob.glob(os.path.join(path_to_models, '1000_ids'))
    
    for ids_dir in ids_dirs:
        ids_dirname = os.path.relpath(ids_dir, path_to_models)
        num_classes = int(ids_dirname[:-4])
        model = ModelInitializer(json.loads(config['MODELLING']['feature_parallelized_architectures'])).get_model(config['MODELLING']['architecture'], False, num_classes)
        for experiment_dir in glob.glob(os.path.join(ids_dir, '*')):
            experiment_dirname =os.path.relpath(experiment_dir, ids_dir)
            num_images = int(experiment_dirname[5 + experiment_dirname.index('_'): -18])
            loc = os.path.join(experiment_dir, 'vgg16', 'models', '119.pth')
            if not os.path.exists(loc):
                loc = os.path.join(experiment_dir, 'vgg16', 'models', '120.pth')
            try:
                model, _1, _2, _3 = model_store.load_model_and_optimizer_loc(model, model_location=loc)

                os.makedirs(config['REP_BEHAVIOUR']['reps_results_path'], exist_ok=True)
                config['REP_BEHAVIOUR']['reps_results_path'] = os.path.join(results_root,
                str(num_classes) + '_' + str(num_images))
                reps_behaviour_extractor = setup_pairs_reps_behaviour(config, image_loader)
                if reps_behaviour_extractor != None:
                    output = reps_behaviour_extractor.compare_lists(model)
                print('Saving results in ', config['REP_BEHAVIOUR']['reps_results_path'])
            except:
                print('Could not run loc')






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


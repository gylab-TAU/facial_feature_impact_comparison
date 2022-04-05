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
from data_prep.random_experiment_image_loader import RandomExperimentImageLoader
from modelling.local_model_store import LocalModelStore
from experiment_setup.lfw_test_setup import get_lfw_test
from experiment_setup.deductive_experiment_dataloaders_setup import dataloaders_setup
from experiment_setup.generic_trainer_setup import get_trainer
from experiment_setup.pairs_behaviour_setup import setup_pairs_reps_behaviour
mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)


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

    num_iterations = int(config['GENERAL']['num_experiments'])
    num_ids = json.loads(config['DATASET']['num_ids'])
    num_pics = json.loads(config['DATASET']['num_pics'])
    overall_index = 0

    all_lfw_results = None
    all_output = None
    experiment_name = config['GENERAL']['experiment_name']


    for id in num_ids:
        config['GENERAL']['root_dir'] = config['GENERAL']['base_root_dir']
        print('before id_folder:',config['GENERAL']['id_folder']  )
        print('base root:',config['GENERAL']['base_root_dir']  )
        config['GENERAL']['id_folder'] = str(id) + '_ids'
        config['GENERAL']['root_dir'] =os.path.join(config['GENERAL']['base_root_dir'],
                                                config['GENERAL']['id_folder'])
        print('after id_folder:',config['GENERAL']['id_folder']  )
        print('base root:',config['GENERAL']['base_root_dir']  )

        #mlflow set experiment
        if mlflow.get_experiment_by_name(config['GENERAL']['experiment_name']) is None:
            mlflow.create_experiment(config['GENERAL']['experiment_name'], artifact_location=os.path.join(const.MLFLOW_ARTIFACT_STORE, config['GENERAL']['experiment_name']))
        mlflow.set_experiment(config['GENERAL']['experiment_name'])
        run_name = None
        if 'run_name' in config['GENERAL']:
            run_name = config['GENERAL']['run_name']


        for pic in num_pics:
            for i in range(num_iterations):
                print("________   ITER   " + str(i) + " ________")
                print("________  NUM IDS " + str(id) + " ________")
                print("________ NUM PICS " + str(pic) + " ________")


                #mlflow set run
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
                    mlflow.log_param('ids', str(id))
                    mlflow.log_param('images', str(pic))
                    mlflow.log_param('index', str(i))

                    crop_scale = None
                    if 'crop_scale' in config['DATASET']:
                        crop_scale = json.loads(config['DATASET']['crop_scale'])
                        crop_scale = (crop_scale['max'], crop_scale['min'])



                    config['GENERAL']['experiment_name'] = experiment_name + '_' + str(id) + '_' + str(pic) + '_' + str(i)
                    print(config['GENERAL']['root_dir'])
                    # Create image loader (by the configuration)
                    im_size = int(config['DATASET']['image_size'])
                    post_crop_im_size = int(config['DATASET']['post_crop_im_size'])
                    dataset_means = json.loads(config['DATASET']['dataset_means'])
                    dataset_stds = json.loads(config['DATASET']['dataset_stds'])
                    image_loader = RandomExperimentImageLoader(im_size, post_crop_im_size, dataset_means, dataset_stds)

                    print("training on dataset: ", config['DATASET']['raw_dataset_path'])

                    # Create dataloader for the training
                    dataloaders = dataloaders_setup(config, config['DATASET']['raw_dataset_path'], image_loader, id, pic)

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

                    # Creating the trainer and loading the pretrained model if specified in the configuration
                    trainer = get_trainer(config, id, start_epoch, lfw_tester)

                    # Will train the model from start_epoch to (end_epoch - 1) with the given dataloaders
                    trainer.train_model(start_epoch, end_epoch, dataloaders)

                    lfw_results = lfw_tester.test_performance(trainer.model)
                    lfw_results['index'] = overall_index
                    lfw_results['num_ids'] = id
                    lfw_results['num_pics'] = pic
                    if all_lfw_results is None:
                        all_lfw_results = lfw_results
                    else:
                        all_lfw_results = all_lfw_results.append(lfw_results)
                    print(lfw_results)

                    reps_behaviour_extractor = setup_pairs_reps_behaviour(config, image_loader)
                    if reps_behaviour_extractor != None:
                        output = reps_behaviour_extractor.compare_lists(trainer.model)
                        output['index'] = overall_index
                        output['num_ids'] = id
                        output['num_pics'] = pic
                        # saving results for the meanwhile
                        results_path = os.path.join(config['REP_BEHAVIOUR']['reps_results_path'],
                                                    config['REP_BEHAVIOUR']['output_filename'] + '.csv')
                        lfw_path = os.path.join(config['REP_BEHAVIOUR']['reps_results_path'], 'logs.csv')
                        os.makedirs(config['REP_BEHAVIOUR']['reps_results_path'], exist_ok=True)
                        output.to_csv(results_path)
                        lfw_results.to_csv(lfw_path)
                        if all_output is None:
                            all_output = output
                        else:
                            all_output = all_output.append(output)
                        mlflow.log_artifact(results_path)


                overall_index += 1
                print(all_output)
                print('all_lfw_results:',all_lfw_results)

    config['GENERAL']['experiment_name'] = experiment_name
    # lfw_path = os.path.join(root_dir, experiment_name + '_all', rep_dir, lfw_filename)
    # results_path = os.path.join(root_dir, experiment_name + '_all', rep_dir, rep_filename)
    results_path = os.path.join(config['REP_BEHAVIOUR']['reps_results_path'],
                                config['REP_BEHAVIOUR']['output_filename'] + '.csv')

    lfw_path = os.path.join(config['REP_BEHAVIOUR']['reps_results_path'], config['LFW_TEST']['output_filename'] + '.csv')
    print('Saving results in ', results_path)
    os.makedirs(config['REP_BEHAVIOUR']['reps_results_path'], exist_ok=True)
    if all_output is not None:
        all_output.to_csv(results_path)
        all_lfw_results.to_csv(lfw_path)


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

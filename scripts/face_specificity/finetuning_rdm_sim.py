import mlflow
import const
import plotly as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score


def get_experiment_val_acc(experiment_name: str):
    exp_id = dict(mlflow.get_experiment_by_name(experiment_name))['experiment_id']
    runs = mlflow.search_runs(exp_id)
    name2val_acc = runs[['start_time', 'tags.mlflow.runName', 'metrics.val acc']]
    name2val_acc = name2val_acc.groupby('tags.mlflow.runName').min('start_time')
    return name2val_acc


def get_rdms_locations():
    index = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    rdms_locations = {
        'objects pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_inanimate_upright_first_after_fc8.csv',
        ],
        'objects finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/29cb5b74c47142f6b4aaf6d0720cffc0/artifacts/30_inanimate_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/52971a4cb9384b66bb392d646417ab22/artifacts/30_inanimate_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e25a2dc1e30d4bc7b79e5f8689c17377/artifacts/30_inanimate_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/68799c15febf4a7db2ac20abdd3b66db/artifacts/30_inanimate_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/7e1b92e1aa6741cdb23fbe44e9902f98/artifacts/30_inanimate_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/704411d7e7f44206b53d44e95fcfd34d/artifacts/30_inanimate_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e6172740497c48359f989fe036616b52/artifacts/30_inanimate_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/a01dd4019e4a46cf9fd9dcdcb5db7b7d/artifacts/30_inanimate_upright_first_after_fc8.csv',
        ],
        'faces pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_faces_upright_first_after_fc8.csv',
        ],
        'faces finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/24439a0c88054d60b7d089402a18d17a/artifacts/30_faces_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/58bc071777184953bfc86798eb01c71c/artifacts/30_faces_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/2e41ba20d78a4e5aaa4153698930b327/artifacts/30_faces_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/7ce069cecde14d98aaa5ee8910403787/artifacts/30_faces_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/8dbb01e7b3ef4c16adc270a4879c1c75/artifacts/30_faces_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b0a4b6af14194a4b8927916f96acb392/artifacts/30_faces_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b59cf78ab7764b1a8685c18a3986c631/artifacts/30_faces_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/5bf470ff62744068adcbd5a831c8dd8b/artifacts/30_faces_upright_first_after_fc8.csv',
        ],
        'individual birds from objects pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
        'individual birds from objects finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/29cb5b74c47142f6b4aaf6d0720cffc0/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/52971a4cb9384b66bb392d646417ab22/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e25a2dc1e30d4bc7b79e5f8689c17377/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/68799c15febf4a7db2ac20abdd3b66db/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/7e1b92e1aa6741cdb23fbe44e9902f98/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/704411d7e7f44206b53d44e95fcfd34d/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e6172740497c48359f989fe036616b52/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/a01dd4019e4a46cf9fd9dcdcb5db7b7d/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
        'bird species from objects pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_fc8.csv',
        ],
        'bird species from objects finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/29cb5b74c47142f6b4aaf6d0720cffc0/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/52971a4cb9384b66bb392d646417ab22/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e25a2dc1e30d4bc7b79e5f8689c17377/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/68799c15febf4a7db2ac20abdd3b66db/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/7e1b92e1aa6741cdb23fbe44e9902f98/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/704411d7e7f44206b53d44e95fcfd34d/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e6172740497c48359f989fe036616b52/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/a01dd4019e4a46cf9fd9dcdcb5db7b7d/artifacts/30_bird_species_upright_first_after_fc8.csv',
        ],
        'individual birds from faces pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
        'individual birds from faces finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/24439a0c88054d60b7d089402a18d17a/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/58bc071777184953bfc86798eb01c71c/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/2e41ba20d78a4e5aaa4153698930b327/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/7ce069cecde14d98aaa5ee8910403787/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/8dbb01e7b3ef4c16adc270a4879c1c75/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b0a4b6af14194a4b8927916f96acb392/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b59cf78ab7764b1a8685c18a3986c631/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/5bf470ff62744068adcbd5a831c8dd8b/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
        'bird species from faces pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_fc8.csv',
        ],
        'bird species from faces finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/24439a0c88054d60b7d089402a18d17a/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/58bc071777184953bfc86798eb01c71c/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/2e41ba20d78a4e5aaa4153698930b327/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/7ce069cecde14d98aaa5ee8910403787/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/8dbb01e7b3ef4c16adc270a4879c1c75/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b0a4b6af14194a4b8927916f96acb392/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b59cf78ab7764b1a8685c18a3986c631/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/5bf470ff62744068adcbd5a831c8dd8b/artifacts/30_bird_species_upright_first_after_fc8.csv',
        ],
    }
    return pd.DataFrame(rdms_locations, index)


# RDMs locations

def get_rdms_locations_full_layer():
    index = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    rdms_locations = {
        'objects pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/95fafb1431d6435f9bf751bc6c31358b/artifacts/30_inanimate_upright_fc8.csv',
        ],
        'objects finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/13c37646efb04f7b99c80dc4ec28170f/artifacts/30_inanimate_upright_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/78743df67d304323adf06fd08c0d792b/artifacts/30_inanimate_upright_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/587878c149404fb09494076b4248d5c4/artifacts/30_inanimate_upright_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/3a7789feb80541c1a242c22bfce2bd7a/artifacts/30_inanimate_upright_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/07918b42809640e1adbc45b92255e8c8/artifacts/30_inanimate_upright_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/73d16510643d48438b7f3de66d2ebfce/artifacts/30_inanimate_upright_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/754107b0cc074781b83445304297265b/artifacts/30_inanimate_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/79fd02e5afec4cf8a402dd62cb67e42c/artifacts/30_inanimate_upright_fc8.csv',
        ],
        'faces pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c40e2d720674457f9c4854d98e770a0d/artifacts/30_faces_upright_fc8.csv',
        ],
        'faces finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/e28472d9a01a4c1fbcbc45f3aa0579e6/artifacts/30_faces_upright_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/a7c28c75638540f9996453c2114cbb1b/artifacts/30_faces_upright_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/20db33062e3b41ea98de94ac4c9c551a/artifacts/30_faces_upright_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/0c01518d6d0240aaabd37c5ef1a2bf52/artifacts/30_faces_upright_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/4b5ea66637494dd6a54eae6486d50161/artifacts/30_faces_upright_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/3fdb19d6d7fe40ea95ab43c6623a15a5/artifacts/30_faces_upright_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/3b27216dd19d45159772f45b99266f78/artifacts/30_faces_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/f7ec4a8f7ff54e07bd99d5c8ac96dd10/artifacts/30_faces_upright_fc8.csv'
        ]
    }
    return pd.DataFrame(rdms_locations, index)


def get_indiv_birds_rdms_locations():
    index = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    rdms_locations = {
        'individual birds from objects pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
        'individual birds from objects finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/29cb5b74c47142f6b4aaf6d0720cffc0/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/52971a4cb9384b66bb392d646417ab22/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e25a2dc1e30d4bc7b79e5f8689c17377/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/68799c15febf4a7db2ac20abdd3b66db/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/7e1b92e1aa6741cdb23fbe44e9902f98/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/704411d7e7f44206b53d44e95fcfd34d/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/e6172740497c48359f989fe036616b52/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/a01dd4019e4a46cf9fd9dcdcb5db7b7d/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
        'individual birds from faces pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
        'individual birds from faces finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/24439a0c88054d60b7d089402a18d17a/artifacts/30_individual_birds_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/58bc071777184953bfc86798eb01c71c/artifacts/30_individual_birds_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/2e41ba20d78a4e5aaa4153698930b327/artifacts/30_individual_birds_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/7ce069cecde14d98aaa5ee8910403787/artifacts/30_individual_birds_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/8dbb01e7b3ef4c16adc270a4879c1c75/artifacts/30_individual_birds_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b0a4b6af14194a4b8927916f96acb392/artifacts/30_individual_birds_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/b59cf78ab7764b1a8685c18a3986c631/artifacts/30_individual_birds_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/5bf470ff62744068adcbd5a831c8dd8b/artifacts/30_individual_birds_upright_first_after_fc8.csv',
        ],
    }
    return pd.DataFrame(rdms_locations, index)


def get_species_rdms_locations():
    index = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    rdms_locations = {
        'bird species from objects pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/0201ce91a686452794b3c81cbb26b1e6/artifacts/30_bird_species_upright_first_after_fc8.csv',
        ],
        'bird species from objects finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/9faceeeb9f18439e8854f76a71336569/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/7d4ab7dff7e5448ea0479742b86befe0/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/2814a6c6698048e996b5e138d80adce3/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/58e2519bb82c4d97b288ed738bcf0275/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/0fcb6b2cd2654813b163d9433251338e/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/84829a50c0b249f69184a2c9e62a9cd6/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/02fd6c683c83433d875a23f5a756b379/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/e34ab2216b1c4a4e856310fd700ac63a/artifacts/30_bird_species_upright_first_after_fc8.csv',
        ],
        'bird species from faces pretrained': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/60997c44b3dc48d7b979b34a1f55c21e/artifacts/30_bird_species_upright_first_after_fc8.csv',
        ],
        'bird species from faces finetuned': [
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/fe50937e5aa34912a59fa9a6449a6331/artifacts/30_bird_species_upright_first_after_conv1.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/344850163fc7479fb0bcdd0e8b046c27/artifacts/30_bird_species_upright_first_after_conv2.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/556d8f455949440cbd9f34a65a3dbd1b/artifacts/30_bird_species_upright_first_after_conv3.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/9d502aae05e447a38c4ac71fc9954827/artifacts/30_bird_species_upright_first_after_conv4.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/0338a9b838bb4ba2a42dbd9efc194795/artifacts/30_bird_species_upright_first_after_conv5.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/264c644943d748ac94f657ec0672c26d/artifacts/30_bird_species_upright_first_after_fc6.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/227e014860274df288f945a1f029c401/artifacts/30_bird_species_upright_first_after_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/78259caa34a840f28c8667a34de85de3/artifacts/30_bird_species_upright_first_after_fc8.csv'
        ],
    }
    return pd.DataFrame(rdms_locations, index)


def get_indiv_birds_rdms_fc7_locations():
    index = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'pretrained']
    rdms_locations = {
        'individual birds from objects': [

        ],
        'individual birds from faces': [

        ],
    }
    return pd.DataFrame(rdms_locations, index)


def get_rdms_fc7_locations():
    index = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'pretrained']
    rdms_locations = {
        'individual birds from objects': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/1aee725c5d9e44cab8ac9d343c42b0d5/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/a8560b22d68645f49c40513aabe22498/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/f8bc8d7fc75041688e029bb31f5082da/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/1981acc046a847ef8d19481a00189314/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/c079834842564cadbe0084c7cc5587dd/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/884c22f1c446428ab4d08c28a82be637/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/19e0d66b839c4c53a59454708ece3f1d/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/452bfc7883b040c4b742e500f968c0b9/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/1f4e3a4af46e46dfb6485c9b6d306f02/artifacts/30_individual_birds_upright_fc7.csv',
        ],
        'individual birds from faces': [
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/a605baffad8f4f29a6085a155eb1d5c9/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c0206b0156dc427faa72c26280a065d4/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/0b9682b9611046d992c377e866378fa7/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/f42de5ba6da04b32977f82bd31f18e62/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/2702b9e87a5e4217b9372e590868f417/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/29b784699455471eb94bf0a3b3156f59/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/c82902685c124d0da7e03b62297f5eb2/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/2aa5b2b2f61c4be2a4f382e4954f92ca/artifacts/30_individual_birds_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/5bb2a413451a400f9032ae769cbdca35/artifacts/30_individual_birds_upright_fc7.csv',
        ],
        'bird species from objects': [
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/6c32610e29314e57984b51c5568bc575/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/35bad510317f47888767168dd226c326/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/02fb1327d85a42249838dc7c99bddbd7/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/d213051e1c2d474a9b3951ec8a129d83/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/7293a7c4a0a948fb97cd66aec8322d60/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/b7461e01c9744371bdd72e68d8c7d4b6/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/0ae41129b1694727b5a43ac02e9fbcfc/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_objects_finetuning/9b1b5e3737664c9b843c2229b5d53718/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_objects_finetuning/1f4e3a4af46e46dfb6485c9b6d306f02/artifacts/30_bird_species_upright_fc7.csv'
        ],
        'bird species from faces': [
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/6c36ca9bfcba414bbdc00cc3c4f4cb95/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/d7f0c9de095648688c12c79981031a3a/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/262f25a928c146ceb10834bffb10e6f3/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/2925765f27c34b49ac87ceb2726375de/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/7c8c8c55500746ec9b3eca22c1124ddd/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/5e4c3d7f7e884828b531ffced3e02737/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/f23b5f8605e84fa59bc21760e5dc3073/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/birds_faces_finetuning/9214c94e1f1543da8dbc91eae37b1212/artifacts/30_bird_species_upright_fc7.csv',
            '/home/hdd_storage/mlflow/artifact_store/individual_birds_faces_finetuning/5bb2a413451a400f9032ae769cbdca35/artifacts/30_bird_species_upright_fc7.csv'
        ],
    }
    return pd.DataFrame(rdms_locations, index)

# END: RDMs locations

def save_rdms_heatmap(rdm_paths: pd.DataFrame):
    titles = []
    for i, layer in enumerate(rdm_paths.index):
        for j, key in enumerate(rdm_paths.columns):
            titles.append(f'{key}, {rdm_paths.iloc[i].index} RDM')
    indiv_rdms = make_subplots(rows=2, cols=9,
                         x_title='Finetuning depth', y_title='Pretraining domain',
                         row_titles=list(rdm_paths.columns[:2]), column_titles=list(rdm_paths.index),
                         )
    species_rdms = make_subplots(rows=2, cols=9,
                               x_title='Finetuning depth', y_title='Pretraining domain',
                               row_titles=list(rdm_paths.columns[2:]), column_titles=list(rdm_paths.index),
                               )#vertical_spacing=0.05

    indiv_rdms.update_layout(height=2000, width=1000, template='plotly_white')
    species_rdms.update_layout(height=2000, width=1000, template='plotly_white')
    for j, col in enumerate(rdm_paths.columns):
        for i, idx in enumerate(rdm_paths.index):
            curr_path = rdm_paths.iloc[i][col]
            print(curr_path)
            df = pd.read_csv(curr_path, index_col='Unnamed: 0')
            title = f'{col}, {idx} RDM'
            row_idx = 1 + (j % 2)
            col_idx = 1 + i
            if j < 2:
                print('indiv_rdms')
                indiv_rdms.add_trace(
                    go.Heatmap(
                        z=df, name=titles[i + j], zmin=0, zmax=1),
                    row=row_idx, col=col_idx)
            else:
                print('species_rdms')
                species_rdms.add_trace(
                    go.Heatmap(
                        z=df, name=titles[i + j], zmin=0, zmax=1),
                    row=row_idx, col=col_idx)
    indiv_rdms.update_layout(template='plotly_white')
    indiv_rdms.update_layout(height=750, width=9 * 750 // 2, template='plotly_white')
    indiv_rdms.write_html(
        '/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/indiv_bird_rdms_fc7.html')
    indiv_rdms.show()

    species_rdms.update_layout(template='plotly_white')
    species_rdms.update_layout(height=750, width=9 * 750 // 2, template='plotly_white')
    species_rdms.write_html(
        '/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/species_bird_rdms_fc7.html')
    species_rdms.show()

    go.Image



def rdm_to_dist_list(rdm: pd.DataFrame):
    rdm = rdm.where(np.triu(np.ones(rdm.shape)).astype(np.bool))
    for i in range(len(rdm)):
        rdm.iloc[i, i] = np.nan
    stacked = rdm.stack()
    no_diag = pd.DataFrame(stacked.dropna()).rename(columns={0: 'cos'})
    print(no_diag.shape)
    return no_diag


def rdms_correlation(pretrain_loc, finetune_loc):
    pretrain = rdm_to_dist_list(pd.read_csv(pretrain_loc, index_col='Unnamed: 0'))
    finetune = rdm_to_dist_list(pd.read_csv(finetune_loc, index_col='Unnamed: 0'))
    # return stats.kendalltau(pretrain, finetune).correlation
    joined = pretrain.join(finetune, lsuffix='_pretrained', rsuffix='_finetuned')
    ktau = stats.kendalltau(joined['cos_pretrained'].to_numpy(), joined['cos_finetuned'].to_numpy())
    print(ktau)
    return ktau.correlation


def locations2corr(rdms_locations):
    """
    Given the locations to the RDMs, calculate the correaltions before and after training
    """
    index = []
    correlations = {'Kendall Tau': [], 'Expertise': []}
    for layer in rdms_locations.index:
        index.append(layer)
        index.append(layer)
        index.append(layer)
        index.append(layer)
        index.append(layer)
        index.append(layer)
        layer_locations = rdms_locations.loc[layer]

        correlations['Kendall Tau'].append(
            rdms_correlation(
                layer_locations['objects pretrained'],
                layer_locations['objects finetuned']))
        correlations['Expertise'].append('Objects')

        correlations['Kendall Tau'].append(
            rdms_correlation(
                layer_locations['faces pretrained'],
                layer_locations['faces finetuned']))
        correlations['Expertise'].append('Faces')

        correlations['Kendall Tau'].append(
            rdms_correlation(
                layer_locations['individual birds from objects pretrained'],
                layer_locations['individual birds from objects finetuned']))
        correlations['Expertise'].append('Individual birds (inanimate backbone)')

        correlations['Kendall Tau'].append(
            rdms_correlation(
                layer_locations['bird species from objects pretrained'],
                layer_locations['bird species from objects finetuned']))
        correlations['Expertise'].append('Bird species (inanimate backbone)')

        correlations['Kendall Tau'].append(
            rdms_correlation(
                layer_locations['individual birds from faces pretrained'],
                layer_locations['individual birds from faces finetuned']))
        correlations['Expertise'].append('Individual birds (faces backbone)')

        correlations['Kendall Tau'].append(
            rdms_correlation(
                layer_locations['bird species from faces pretrained'],
                layer_locations['bird species from faces finetuned']))
        correlations['Expertise'].append('Bird species (faces backbone)')
    df = pd.DataFrame(correlations, index=index)
    df.index.name = 'Layer'
    return df


def rdm_to_verification_dist_list(rdm: pd.DataFrame):
    """
    Given an RDM, get a list of distance
    Removes duplicate pairs
    Removes the main diagonal (distance between an image to itself
    """
    rdm.index = [str(i // 10) + ':' + str(col) for i, col in enumerate(rdm.index)]
    rdm.columns = [str(i // 10) + ':' + str(col) for i, col in enumerate(rdm.columns)]
    for i in range(len(rdm)):
        rdm.iloc[i, i] = np.nan
    stacked = rdm.stack()
    no_diag = pd.DataFrame(stacked.dropna()).rename(columns={0: 'cos'})

    same = []
    for idx in no_diag.index:
        same.append(idx[0].split(':')[0] == idx[1].split(':')[0])
    no_diag['same'] = same
    return no_diag


def calc_graph_measurements(df: pd.DataFrame, label_col: str, value_col: str):
    """ Calculate the fpr, tpr, thresholds, and AUC for them """
    fpr, tpr, thresh = roc_curve(df[label_col], df[value_col], pos_label=0)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresh, roc_auc


def plot_rocs(rdm_paths):
    """ Plot the ROCs for the RDMs """
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
        dict(color='royalblue', width=2, dash='dot'),
        dict(color='firebrick', width=2, dash='dot'),
    ]
    titles = []
    for i in range(9):
        for j, key in enumerate(rdm_paths.columns):
            titles.append(f'{key}, {rdm_paths.index[i]} ROC')

    rocs = make_subplots(rows=2, cols=9,
                         x_title='Finetuning depth', y_title='Pretraining domain',
                         row_titles=['Individual birds verification', 'Bird species verification'], column_titles=list(rdm_paths.index))
                         # vertical_spacing=0.5)

    for i, idx in enumerate(rdm_paths.index):
        for j, col in enumerate(rdm_paths.columns):
            curr_path = rdm_paths.iloc[i][col]
            df = pd.read_csv(curr_path, index_col='Unnamed: 0')
            verification_dist_list = rdm_to_verification_dist_list(df)
            fpr, tpr, thresh, roc_auc = calc_graph_measurements(verification_dist_list, 'same', 'cos')
            name = f'{col}, {idx}'
            rocs.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'{name}. AUC={roc_auc}', mode='lines', line=line_designs[j % 2]),
                row=1 + (j // 2), col=1 + i)
    for i in range(9):
        for j in range(2):
            rocs.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1, row=1 + j, col=1 + i)

    rocs.update_layout(height=750, width=9*750//2, template='plotly_white')
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.show()
    rocs.write_html('/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/verification_over_finetune_depth_16022022.html')


def plot_aucs(rdm_paths):
    """ Create line plot of AUC as function of finetuning depth """
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
        dict(color='royalblue', width=2, dash='dot'),
        dict(color='firebrick', width=2, dash='dot'),
    ]
    aucs = {}
    for j, col in enumerate(rdm_paths.columns):
        aucs[col] = []
        for i, idx in enumerate(rdm_paths.index):
            curr_path = rdm_paths.iloc[i][col]
            df = pd.read_csv(curr_path, index_col='Unnamed: 0')
            verification_dist_list = rdm_to_verification_dist_list(df)
            fpr, tpr, thresh, roc_auc = calc_graph_measurements(verification_dist_list, 'same', 'cos')
            aucs[col].append(roc_auc)
    aucs = pd.DataFrame(aucs, index = rdm_paths.index)
    fig = go.Figure()
    for i, col in enumerate(aucs.columns):
        fig.add_trace(go.Scatter(x=aucs.index, y=aucs[col],
                                 mode='lines',
                                 name=col, line=line_designs[i]))

    fig.update_layout(
        title="Verification AUC as function of finetuning depth",
        xaxis_title="Finetune depth",
        yaxis_title="Verification AUC",
        legend_title="Legend",
        template='plotly_white'
    )
    fig.update_yaxes(range=[0.0, 1.0])

    fig.show()
    aucs.to_csv('/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/verification_AUC_over_finetune_depth_16022022.csv')
    fig.write_html('/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/verification_AUC_over_finetune_depth_range01_16022022.html')



def plot_correlation(correlations):
    """ Plot the correlations between RDMs """
    fig = px.line(correlations, x=correlations.index, y='Kendall Tau', color='Expertise')
    fig.update_yaxes(range=[0.0, 1.0])
    fig.write_html('/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/ktau_pre-post_fineuning_with_birds.html')
    fig.show()


def save_files(rdm_locations):
    """ Move the RDM files from the original storage (in MLFlow) to a more convenient location"""
    for idx in rdm_locations.index:
        for col in rdm_locations.columns:
            curr_path = rdm_locations.loc[idx, col]
            df = pd.read_csv(curr_path, index_col='Unnamed: 0')
            df.to_csv(f'/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/rdm_data/{col}, {idx}.csv')


if __name__ == '__main__':
    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    # locations = get_rdms_locations()
    # correlations = locations2corr(locations)
    # plot_correlation(correlations)
    # locations = get_indiv_birds_rdms_locations()
    locations = get_rdms_fc7_locations()
    # save_rdms_heatmap(locations)
    plot_rocs(locations)
    plot_aucs(locations)
    # save_files(locations)

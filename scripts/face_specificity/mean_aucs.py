from argparse import ArgumentParser
from typing import Dict, List, Tuple

from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors


def get_rdms_locations() -> Dict[str,List[str]]:
    locs = {
        'sociable weavers objects backbone':
            [
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/0ed4376fa3154db69c6b950294fba15e/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/8adc7fe9426a4b0b8b60b5ec7d2ae152/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/bf4b77b00526411f8c68c353b7ab4ee8/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/isociable_weavers_objects_finetuning/82290825ab614e9b85ca9255a42a0106/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/137ff53cc643431888571444b872d73b/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/5e86ba879c38405491c7b4aec3f913ff/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/9d13b4beb69d48a3b9b4d1e83be1d1c4/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/2fcadc642a4d404dafe3d14bc3ecf3f9/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/bc671ffbd43a482c9e7da1b7de9e8d13/artifacts/30_individual_birds_upright_fc7.csv'
            ],
        'sociable weavers faces backbone':
            [
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/6e366e679c754049adfd2d5d3e22c48c/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/49391d6c36694434b154bebb0d5a7fd1/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/8397d758a2654a828399ce4287342efc/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/3b59641ead4e4005bdca36d7000af0a1/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/7c15f70664a54a38a41fef9ce5db1130/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/feb44f895e454e01997f8fe547eb8a9d/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/feef7c54561f4fd59e9e035cecb52dbb/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/5c7262256c274c40b1642c51a1c27961/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/a1ce2bd8ccf04836bbbd5704a24a8703/artifacts/30_individual_birds_upright_fc7.csv'
            ],
        'bird species objects backbone':
            [
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
        'bird species faces backbone':
            [
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
    return locs


def load_rdm(loc: str) -> pd.DataFrame:
    """Given a path to a csv RDM, loads it"""
    return pd.read_csv(loc, index_col=0)


def remove_rdm_redundancies(rdm:pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records from the RDM (choose only upper triangle)
    Remove distances from an item to itself (remove the main diagonal)
    """
    temp = rdm.copy()
    lt = np.tril(np.ones(rdm.shape), -1).astype(np.bool)

    temp = temp.where(lt == False, np.nan)
    np.fill_diagonal(temp.values, np.nan)
    return temp


def rdm_to_dist_list(rdm: pd.DataFrame) -> pd.DataFrame:
    """
    Given a full RDM, remove redundancies and return it in a format of a distance list
    """
    rdm = remove_rdm_redundancies(rdm)
    rdm.index = [str(i // 10) + ':' + str(col) for i, col in enumerate(rdm.index)]
    rdm.columns = [str(i // 10) + ':' + str(col) for i, col in enumerate(rdm.columns)]
    stacked = rdm.stack()
    no_diag = pd.DataFrame(stacked.dropna()).rename(columns={0: 'cos'})

    same = []
    for idx in no_diag.index:
        same.append(idx[0].split(':')[0] == idx[1].split(':')[0])
    no_diag['same'] = same
    return no_diag


def sample_dist_list(dist_list: pd.DataFrame, n_batches: int = 30) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample n_batches of equal size from the positive distances, and the negative distances
    All negative samples should have the same size as the positive samples
    """
    positive = dist_list[dist_list['same']]
    negative = dist_list[~dist_list['same']]

    positive_shuffled = positive.sample(frac=1)
    negative_shuffled = negative.sample(frac=1)

    split_pos  = np.array_split(positive_shuffled, n_batches)
    n_rows = len(split_pos[0])
    sampled_neg = negative_shuffled.sample(n_rows * n_batches)
    split_neg = np.array_split(sampled_neg, n_batches)

    for i in range(n_batches):
        split_pos[i] = split_pos[i]['cos'].to_numpy()
        split_neg[i] = split_neg[i]['cos'].to_numpy()

    return split_pos, split_neg


def get_rdm_aucs(positives: List[np.ndarray], negatives: List[np.ndarray], n_batches: int):
    aucs = []

    for i in range(n_batches):
        # Mark labels
        pos_labels = np.ones_like(negatives[i])
        neg_labels = np.zeros_like(positives[i])
        labels = np.concatenate([neg_labels, pos_labels])

        # concatenate dists
        dists = np.concatenate([positives[i], negatives[i]])

        # Calculate AUCs
        fpr, tpr, thresholds = metrics.roc_curve(labels, dists, pos_label=1)
        aucs.append(metrics.auc(fpr, tpr))

    return aucs


def plot_raw(faces_aucs: List[List[float]], objs_aucs: List[List[float]], layers: List[str], ds: str) -> None:
    objs_colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')
    faces_colors = n_colors('rgb(5, 5, 200)', 'rgb(200, 200, 10)', 12, colortype='rgb')

    fig = go.Figure()
    for i, (faces_data_line, objs_data_line, obj_color, faces_color, layer) in enumerate(zip(faces_aucs, objs_aucs, objs_colors, faces_colors, layers)):
        fig.add_trace(go.Violin(x=objs_data_line, line_color=obj_color, name=layer, side='positive', text=f'{ds} objects'))
        fig.add_trace(go.Violin(x=faces_data_line, line_color=faces_color, name=layer, side='negative', text=f'{ds} faces'))

    fig.update_traces(orientation='h', points=False) #side='positive', width=3,
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    fig.show()
    fig.write_html(f'/home/ssd_storage/experiments/sociable_weavers/vgg16/results/{ds}_violins_aucs_over_layers.html')


def auc_stats(aucs: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    mean = np.mean(aucs)
    n = len(aucs)
    se = stats.sem(aucs)
    h = se * stats.t.ppf((1+confidence) / 2., n-1)
    return mean, h


def make_trend(aucs: List[List[float]]) -> Tuple[List[float], List[float]]:
    means = []
    err_bar = []
    for layer in aucs:
        m, h = auc_stats(layer)
        means.append(m)
        err_bar.append(h)
    return means, err_bar


def plot_means(faces_aucs: List[List[float]], objs_aucs: List[List[float]], layers: List[str], ds: str) -> None:
    fig = go.Figure()

    face_means, face_err = make_trend(faces_aucs)
    objs_means, objs_err = make_trend(objs_aucs)

    fig.add_trace(go.Scatter(x=layers, y=face_means, name=f'{ds} Faces backbone',
                             error_y=dict(
                                 type='data',
                                 array=face_err,
                                 visible=True)))
    fig.add_trace(go.Scatter(x=layers, y=objs_means, name=f'{ds} Inanimate backbone',
                             error_y=dict(
                                 type='data',
                                 array=objs_err,
                                 visible=True)))
    fig.show()
    fig.write_html(f'/home/ssd_storage/experiments/sociable_weavers/vgg16/results/{ds}_averge_aucs_over_layers.html')


def save_aucs(aucs: List[List[float]], layers: List[str], backbone: str, ds: str) -> None:
    df = pd.DataFrame(aucs, index=layers)
    df.to_csv(f'/home/ssd_storage/experiments/sociable_weavers/vgg16/results/{ds}_random_sampled_{backbone}_aucs.csv')


if __name__ == '__main__':
    locs = get_rdms_locations()
    for ds in ['sociable weavers', 'bird species']:
        dfs = []
        n_batches = 30
        objs_aucs = []
        faces_aucs = []

        for pth in locs[f'{ds} objects backbone']:
            df = load_rdm(pth)
            dist_list = rdm_to_dist_list(df)
            pos, neg = sample_dist_list(dist_list, n_batches)
            aucs = get_rdm_aucs(pos, neg, n_batches)
            objs_aucs.append(aucs)

        for pth in locs[f'{ds} faces backbone']:
            df = load_rdm(pth)
            dist_list = rdm_to_dist_list(df)
            pos, neg = sample_dist_list(dist_list, n_batches)
            aucs = get_rdm_aucs(pos, neg, n_batches)
            faces_aucs.append(aucs)

        layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8', 'Pretrained']
        plot_raw(faces_aucs, objs_aucs, layers, ds)
        plot_means(faces_aucs, objs_aucs, layers, ds)
        save_aucs(objs_aucs, layers, 'objects', ds)
        save_aucs(faces_aucs, layers, 'faces', ds)

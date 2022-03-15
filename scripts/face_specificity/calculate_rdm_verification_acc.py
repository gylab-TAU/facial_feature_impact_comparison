import pandas as pd
from glob import glob
import numpy as np
import torch

def get_paths():
    index = [
        '30_inanimate_upright',
        '30_inanimate_inverted',
        '30_bird_species_upright',
        '30_bird_species_inverted',
        '30_faces_upright',
        '30_faces_inverted',
        '30_individual_birds_upright',
        '30_individual_birds_inverted'
    ]
    paths = {
        'Inanimate model':[
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_inanimate_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_inanimate_inverted/fc7.csv',
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_bird_species_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_bird_species_inverted/fc7.csv',
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_faces_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_faces_inverted/fc7.csv',
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_individual_birds_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_objects_no_animals/vgg16/results/30_individual_birds_inverted/fc7.csv',],

        'Bird species': ['/home/ssd_storage/experiments/birds_invert/vgg16/results/30_inanimate_upright/fc7.csv',
        '/home/ssd_storage/experiments/birds_invert/vgg16/results/30_inanimate_inverted/fc7.csv',
        '/home/ssd_storage/experiments/birds_invert/vgg16/results/30_bird_species_upright/fc7.csv',
        '/home/ssd_storage/experiments/birds_invert/vgg16/results/30_bird_species_inverted/fc7.csv',
        '/home/ssd_storage/experiments/birds_invert/vgg16/results/30_faces_upright/fc7.csv',
        '/home/ssd_storage/experiments/birds_invert/vgg16/results/30_faces_inverted/fc7.csv',
        '/home/ssd_storage/experiments/birds_invert/vgg16/results/30_individual_birds_upright/fc7.csv',
        '/home/ssd_storage/experiments/birds_invert/vgg16/results/30_individual_birds_inverted/fc7.csv',],

        'Faces': ['/home/ssd_storage/experiments/260_faces/vgg16/results/30_inanimate_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_faces/vgg16/results/30_inanimate_inverted/fc7.csv',
        '/home/ssd_storage/experiments/260_faces/vgg16/results/30_bird_species_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_faces/vgg16/results/30_bird_species_inverted/fc7.csv',
        '/home/ssd_storage/experiments/260_faces/vgg16/results/30_faces_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_faces/vgg16/results/30_faces_inverted/fc7.csv',
        '/home/ssd_storage/experiments/260_faces/vgg16/results/30_individual_birds_upright/fc7.csv',
        '/home/ssd_storage/experiments/260_faces/vgg16/results/30_individual_birds_inverted/fc7.csv',],

        'Max imgs faces': ['/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_inanimate_upright.csv',
        '/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_inanimate_inverted.csv',
        '/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_bird_species_upright.csv',
        '/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_bird_species_inverted.csv',
        '/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_faces_upright.csv',
        '/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_faces_inverted.csv',
        '/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_individual_birds_upright.csv',
        '/home/ssd_storage/experiments/vgg19_max_imgs_faces/results/30_individual_birds_inverted.csv',],

        'Sociable weavers': ['/home/ssd_storage/experiments/sociable_weavers_keras/results/30_inanimate_upright.csv',
        '/home/ssd_storage/experiments/sociable_weavers_keras/results/30_inanimate_inverted.csv',
        '/home/ssd_storage/experiments/sociable_weavers_keras/results/30_bird_species_upright.csv',
        '/home/ssd_storage/experiments/sociable_weavers_keras/results/30_bird_species_inverted.csv',
        '/home/ssd_storage/experiments/sociable_weavers_keras/results/30_faces_upright.csv',
        '/home/ssd_storage/experiments/sociable_weavers_keras/results/30_faces_inverted.csv',
        '/home/ssd_storage/experiments/sociable_weavers_keras/results/30_individual_birds_upright.csv',
        '/home/ssd_storage/experiments/sociable_weavers_keras/results/30_individual_birds_inverted.csv',],
    }

    return pd.DataFrame(paths, index=index)


def rdm_to_dist_list(rdm: pd.DataFrame):
    rdm = rdm.where(np.triu(np.ones(rdm.shape)).astype(np.bool))
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


def individual_birds_rdm_to_dist_list(rdm: pd.DataFrame):
    rdm = rdm.where(np.triu(np.ones(rdm.shape)).astype(np.bool))
    for i in range(len(rdm)):
        rdm.iloc[i, i] = np.nan
    stacked = rdm.stack()
    no_diag = pd.DataFrame(stacked.dropna()).rename(columns={0: 'cos'})

    same = []
    for idx in no_diag.index:
        same.append(idx[0] == int(idx[1].split('.')[0]))
    no_diag['same'] = same
    # print(no_diag.groupby('same').count())
    return no_diag


def get_verification_accuracy(dists: torch.Tensor, labels: torch.Tensor):
    pos_dists, _ = torch.sort(dists[labels == 1])
    neg_dists, _ = torch.sort(dists[labels == 0][:pos_dists.shape[0]])
    pos_thresh = (pos_dists[:, None] <= pos_dists).float()
    neg_thresh = (neg_dists[:, None] > pos_dists).float()
    all_thresh = torch.cat((pos_thresh, neg_thresh), dim=0)
    accuracies = all_thresh.mean(dim=0)
    best_thresh_idx = torch.argmax(accuracies)
    best_acc = accuracies[best_thresh_idx]
    best_thresh = pos_dists[best_thresh_idx]
    return best_acc.item(), best_thresh.item()


def get_acc_dataframe(rdms_paths: pd.DataFrame):
    acc_index = [
        [], []
    ]

    accs = {'Accuracy': [], 'Threshold': []}
    for idx in rdms_paths.index:
        for col in rdms_paths.columns:
            acc_index[0].append(idx)
            acc_index[1].append(col)
            rdm_path = rdms_paths.loc[idx, col]
            rdm = pd.read_csv(rdm_path, index_col=0)
            if col in ['Max imgs faces', 'Sociable weavers']:
                pairs = individual_birds_rdm_to_dist_list(rdm)
            else:
                pairs = rdm_to_dist_list(rdm)
            dists = torch.tensor(pairs['cos'].values)
            dists = dists.cuda(non_blocking=True)
            labels = torch.tensor(pairs['same'].values)
            labels = labels.cuda(non_blocking=True)
            rdm_acc, rdm_thresh = get_verification_accuracy(dists, labels)
            accs['Accuracy'].append(rdm_acc)
            accs['Threshold'].append(rdm_thresh)
    pd.DataFrame(accs, index=acc_index).to_csv('/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/all_verification.csv')


if __name__ == '__main__':
    df = get_paths()
    get_acc_dataframe(df)

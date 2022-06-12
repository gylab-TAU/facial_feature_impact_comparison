import os
import numpy
import pandas as pd
from scipy import spatial
import torch
import torchvision
from modelling.factories.arcface.arcface_model_initializier import ArcFaceModelInitializer
from modelling.local_model_store import LocalModelStore
import tqdm


def get_dataset_idx_to_cls(dataset, delta):
    idx_to_dataset = {}
    for cl in tqdm.tqdm(dataset.class_to_idx, desc='cls to idx'):
        idx_to_dataset[dataset.class_to_idx[cl]+ delta] = cl
    return idx_to_dataset


def get_full_dataset_idx_to_cls(paths):
    delta = 0
    idx_to_cls = {}
    for path in paths:
        print(path)
        dataset = torchvision.datasets.ImageFolder(path)
        dataset_idx_cls = get_dataset_idx_to_cls(dataset, delta)
        idx_to_cls = {
            **idx_to_cls,
            **dataset_idx_cls
        }
        delta += len(dataset.classes)
    print('Done with datasets')
    return idx_to_cls


def calculate_matrix_column_distances_gpu(mat: torch.Tensor, idx_to_cls):
    mat.requires_grad = False
    mat = mat / mat.norm(dim=1, p=2)[:, None]
    res = torch.mm(mat, mat.transpose(0, 1))
    res = res.cpu().detach().numpy()
    print(res.shape)
    return pd.DataFrame(res,
                        columns=[idx_to_cls[i] for i in range(res.shape[0])],
                        index=[(i, idx_to_cls[i]) for i in range(res.shape[0])])


def calculate_matrix_column_distances(mat: numpy.ndarray, idx_to_cls):
    mat = mat.cpu().detach().numpy()
    print(mat.shape)
    # initialize columns
    dist_mat = {'cls': [], 'idx': []}
    for idx in idx_to_cls:
        dist_mat[idx_to_cls[idx]] = []

    # for every row
    for i in tqdm.tqdm(range(mat.shape[0])):
        dist_mat['idx'].append(i)
        dist_mat['cls'].append(idx_to_cls[i])
        # for every column
        for j in range(mat.shape[0]):
            # calculate distance
            dist_mat[idx_to_cls[j]].append(spatial.distance.cosine(mat[i], mat[j]))
    return pd.DataFrame(dist_mat)


def check_synapse_distances(architecture, experiment_name):
    pretraining_dataset_path = "/home/administrator/experiments/familiarity/dataset/processed_pretraining_dataset/phase_perc_size/pretraining_fixed_{'train': 0.7, 'val': 0.2, 'test': 0.1}/train"
    finetuning_dataset_path = "/home/administrator/experiments/familiarity/dataset/processed_finetuning_dataset/phase_perc_size/pretraining_fixed_A_{'train': 220, 'val': 50, 'test': 50}/train"
    epoch = 119

    root_dir = '/home/administrator/experiments/familiarity/'
    datasets_paths = [pretraining_dataset_path]
    idx_to_cls = get_full_dataset_idx_to_cls(datasets_paths)

    model = ArcFaceModelInitializer(['vgg16']).get_model('vgg16', False, num_classes=len(idx_to_cls))
    model_store = LocalModelStore(architecture, experiment_name, root_dir)

    stuff = model_store.load_model_and_optimizer(model, epoch=epoch)

    model = stuff[0]
    fc8 = model.classifier[-1]
    dist_mat = calculate_matrix_column_distances_gpu(fc8.weight, idx_to_cls)
    dist_mat.to_csv(os.path.join(root_dir, experiment_name, architecture, 'results', 'fc8_dist_mat.csv'))


if __name__=='__main__':
    check_synapse_distances('vgg16', 'pretraining')

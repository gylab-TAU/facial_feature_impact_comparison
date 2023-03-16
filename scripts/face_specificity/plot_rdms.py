from typing import List, Dict

from utils.plots import plot_rdms
from utils.rdms import load_rdm

def get_rdms_locations() -> Dict[str,List[str]]:
    locs = {
        'Objects pretrained':
            [
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/0ed4376fa3154db69c6b950294fba15e/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/8adc7fe9426a4b0b8b60b5ec7d2ae152/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/bf4b77b00526411f8c68c353b7ab4ee8/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/isociable_weavers_objects_finetuning/82290825ab614e9b85ca9255a42a0106/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/137ff53cc643431888571444b872d73b/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/5e86ba879c38405491c7b4aec3f913ff/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/9d13b4beb69d48a3b9b4d1e83be1d1c4/artifacts/30_individual_birds_upright_fc7.csv',
                # '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/2fcadc642a4d404dafe3d14bc3ecf3f9/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_objects_finetuning/bc671ffbd43a482c9e7da1b7de9e8d13/artifacts/30_individual_birds_upright_fc7.csv'
            ],
        'Faces pretrained':
            [
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/6e366e679c754049adfd2d5d3e22c48c/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/49391d6c36694434b154bebb0d5a7fd1/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/8397d758a2654a828399ce4287342efc/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/3b59641ead4e4005bdca36d7000af0a1/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/7c15f70664a54a38a41fef9ce5db1130/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/feb44f895e454e01997f8fe547eb8a9d/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/feef7c54561f4fd59e9e035cecb52dbb/artifacts/30_individual_birds_upright_fc7.csv',
                # '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/5c7262256c274c40b1642c51a1c27961/artifacts/30_individual_birds_upright_fc7.csv',
                '/home/hdd_storage/mlflow/artifact_store/sociable_weavers_faces_finetuning/a1ce2bd8ccf04836bbbd5704a24a8703/artifacts/30_individual_birds_upright_fc7.csv'
            ]
    }
    return locs


if __name__ == '__main__':
    locs = get_rdms_locations()
    dfs = []
    titles = []
    rows = ([0] * 8) + ([1] * 8)
    cols = ([i for i in range(8)] * 2)
    row_titles = []
    col_titles = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'Pretrained']
    for key in locs:
        row_titles.append(key)
        for layer in col_titles:
            titles.append(f'{key} {layer}')
        for pth in locs[key]:
            dfs.append(load_rdm(pth))
    fig = plot_rdms(dfs, rows, cols, row_titles, col_titles)
    fig.show()
    fig.write_html('/home/ssd_storage/experiments/sociable_weavers/vgg16/results/RDMs.html')

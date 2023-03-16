from glob import glob
from os import path
from typing import Dict
import json


def create_json(labels2paths: Dict[str, str], labels2dirs: Dict[str, str]):
    pairs = {}
    dirs = {}
    for label in labels2paths:
        pairs_lists = glob(path.join(labels2paths[label], '*.csv'))
        for lst in pairs_lists:
            idx = path.splitext(path.basename(lst))[0]
            key = f'{label}_{idx}'
            pairs[key] = lst
            dirs[key] = labels2dirs[label]

    print(json.dumps(pairs))
    print(json.dumps(dirs))


if __name__ == '__main__':
    datasets = {"species_upright": "/home/ssd_storage/datasets/processed/verification_datasets/bird_species",
                "sociable_weavers_upright": "/home/ssd_storage/datasets/processed/phase_perc_size/individual_birds_single_species_{'train': 0.8, 'val': 0.2}/val",
                "faces_upright": "/home/ssd_storage/datasets/processed/30_max_imgs_vggface2_mtcnn white_list_{'train': 0.8, 'val': 0.2}/val",
                "inanimate_objects_upright": "/home/ssd_storage/datasets/processed/num_classes/30_cls_inanimate_imagenet/val",
                "species_inverted": "//home/ssd_storage/datasets/processed/verification_datasets/inverted/species_inverted",
                "sociable_weavers_inverted": "/home/ssd_storage/datasets/processed/verification_datasets/inverted/sociable_weavers_inverted",
                "faces_inverted": "/home/ssd_storage/datasets/processed/verification_datasets/inverted/faces_inverted",
                "inanimate_objects_inverted": "/home/ssd_storage/datasets/processed/verification_datasets/inverted/inanimate_objects_inverted"
                }

    lists_pth = "/home/ssd_storage/experiments/Expertise/verification_pairs_lists/final_form"
    print({key: path.join(lists_pth, key) for key in datasets})
    create_json({key: path.join(lists_pth, key) for key in datasets}, datasets)

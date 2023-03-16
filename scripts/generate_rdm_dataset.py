from data_prep.multi_stage_processing import MultiStageProcessor
from data_prep.num_classes_processing import NumClassProcessor
from data_prep.phase_size_processing import PhaseSizeProcessor
import os
from argparse import ArgumentParser


def process(num_classes: int, output_dataset_dir: str, num_images: int, src_dataset_dir: str, dataset_name: str):
    processor = MultiStageProcessor([
        NumClassProcessor(num_classes, num_classes, os.path.join(output_dataset_dir, 'num_classes')),
        PhaseSizeProcessor(os.path.join(output_dataset_dir, f'random_{num_images}_imgs_from_{num_classes}_cls'), {'images': num_images})])

    return processor.process_dataset(src_dataset_dir, dataset_name)[0]


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--num_cls', type=int, default=30)
    parser.add_argument('--num_imgs', type=int, default=10)
    parser.add_argument('--output_dataset_dir', type=str, default='/home/ssd_storage/datasets/Expertise/RDMs/')
    # parser.add_argument('--src_dataset_dir', type=str, default="/home/KAD/project/datasets/processed/phase_perc_size/260_birds_consolidated_{'train': 0.8, 'val': 0.2}/val")
    parser.add_argument('--dataset_name', type=str, default='familiar_birds_rdm')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # dirs = {
    #     # 'familiar_bird_species_rdm_30_10': "/home/KAD/project/datasets/processed/phase_perc_size/260_birds_consolidated_{'train': 0.8, 'val': 0.2}/val",
    #     # 'familiar_individual_birds_rdm_30_10': "/home/ssd_storage/datasets/processed/phase_perc_size/only_30_class_species",
    #     # 'familiar_faces_rdm_30_10': "/home/ssd_storage/datasets/processed/num_classes/faces_260_num-classes_260/val",
    #     # 'familiar_inanimate_rdm_30_10': "/home/ssd_storage/datasets/processed/num_classes/260_inanimate_imagenet_num-classes_260/val",
    #     'familiar_max_img_faces_rdm_30_10': "/home/ssd_storage/datasets/processed/30_max_imgs_vggface2_mtcnn white_list_{'train': 0.9, 'val': 0.1}/val"
    # }

    dirs = {
        'species_rdm_30_10': '/home/ssd_storage/datasets/processed/30_class_validation_balanced_bird_species/val',
        'sociable_weavers_rdm_30_10': "/home/ssd_storage/datasets/processed/phase_perc_size/individual_birds_single_species_{'train': 0.8, 'val': 0.2}/val",
        'faces_rdm_30_10': "/home/ssd_storage/datasets/processed/30_max_imgs_vggface2_mtcnn white_list_{'train': 0.8, 'val': 0.2}/val",
        'inanimate_objects_rdm_30_10': '/home/ssd_storage/datasets/processed/num_classes/30_cls_inanimate_imagenet/val',
    }
    for ds_name in dirs:
        print(f'{ds_name}:')
        print(process(args.num_cls, args.output_dataset_dir, args.num_imgs, dirs[ds_name], ds_name))
    # print(process(args.num_cls, args.output_dataset_dir, args.num_imgs, args.src_dataset_dir, args.dataset_name))

# python scripts/generate_rdm_dataset.py --src_dataset_dir /home/KAD/project/datasets/processed/phase_perc_size/260_birds_consolidated_{'train': 0.8, 'val': 0.2}/val --dataset_name familiar_bird_species_rdm_30_10
# python scripts/generate_rdm_dataset.py --src_dataset_dir /home/ssd_storage/datasets/processed/phase_perc_size/only_30_class_species --dataset_name familiar_individual_birds_rdm_30_10
# python scripts/generate_rdm_dataset.py --src_dataset_dir /home/ssd_storage/datasets/processed/num_classes/faces_260_num-classes_260/val --dataset_name familiar_faces_rdm_30_10
# python scripts/generate_rdm_dataset.py --src_dataset_dir /home/ssd_storage/datasets/processed/num_classes/260_inanimate_imagenet_num-classes_260/val --dataset_name familiar_inanimate_rdm_30_10
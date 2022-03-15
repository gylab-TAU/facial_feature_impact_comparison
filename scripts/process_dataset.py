from data_prep.WhiteListProcessor import WhiteListProcessor
from data_prep.phase_percentage_processing import PhasePercentageProcessor
from data_prep.multi_stage_processing import MultiStageProcessor
from argparse import ArgumentParser

import pandas as pd


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--source_dataset',
                        default="/home/ssd_storage/datasets/processed/whales_10_plus_images_{'train': 0.5, 'val': 0.5}/train/individuals_with_10_imgs_or_more",
                        type=str, help='Path to src dataset')
    parser.add_argument('--dataset_name', default='whales_10_plus_images_triplet', type=str,
                        help='Path to src dataset')
    parser.add_argument('--train_perc', default=0.7, type=float,
                        help='Percent of samples for train dataset')
    parser.add_argument('--val_perc', default=0.3, type=float,
                        help='Percent of samples for validation dataset')
    parser.add_argument('--test_perc', default=0.0, type=float,
                        help='Percent of samples for test dataset')
    parser.add_argument('--white_list_path', default='/home/ssd_storage/experiments/vgg19_faces/top_img_cls.csv', type=str,
                        help='Path to white list file')
    parser.add_argument('--img_dir_is_cls_train', action='store_false',
                        help='Path to white list file')
    parser.add_argument('--output_dataset_dir', default='/home/ssd_storage/datasets/processed', type=str,
                        help='Path to white list file')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    whitelist = pd.read_csv(args.white_list_path)
    whitelist = list(whitelist['class'])
    print(whitelist)
    if args.test_perc > 0:
        phase_percentage_dict = {'train': args.train_perc, 'val': args.val_perc, 'test': args.test_perc}
    else:
        phase_percentage_dict = {'train': args.train_perc, 'val': args.val_perc}
    proc = MultiStageProcessor([
        # WhiteListProcessor(output_dataset_dir=args.output_dataset_dir, white_list=whitelist, img_dir_is_cls=args.img_dir_is_cls_train),
        PhasePercentageProcessor(output_dataset_dir=args.output_dataset_dir, phase_percentage_dict=phase_percentage_dict)
    ])

    print(proc.process_dataset(raw_dataset_dir=args.source_dataset, dataset_name=args.dataset_name))
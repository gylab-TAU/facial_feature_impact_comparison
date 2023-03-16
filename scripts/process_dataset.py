import sys
sys.path.append('/home/administrator/PycharmProjects/libi/facial_feature_impact_comparison')

from data_prep.WhiteListProcessor import WhiteListProcessor
from data_prep.phase_percentage_processing import PhasePercentageProcessor
from data_prep import PhaseSizeProcessor, ClassSizeProcessing
from data_prep.num_classes_processing import NumClassProcessor
from data_prep.multi_stage_processing import MultiStageProcessor
from argparse import ArgumentParser

import pandas as pd


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--source_dataset',
                        # default="/home/ssd_storage/datasets/processed/whales_10_plus_images_{'train': 0.5, 'val': 0.5}/train/individuals_with_10_imgs_or_more",
                        # default="/home/ssd_storage/datasets/vggface2_mtcnn",
                        # default="/home/ssd_storage/datasets/processed/260_faces_vggface_idx_num-classes_260",
                        default="/home/ssd_storage/datasets/students/context/training",
                        type=str, help='Path to src dataset')
    parser.add_argument('--size_or_perc', default='size', type=str, help='size / perc')
    parser.add_argument('--dataset_name', default='youtube_faces', type=str,
                        help='Path to src dataset')
    parser.add_argument('--train_perc', default=0.7, type=float,
                        help='Percent of samples for train dataset')
    parser.add_argument('--val_perc', default=0.2, type=float,
                        help='Percent of samples for validation dataset')
    parser.add_argument('--test_perc', default=0.1, type=float,
                        help='Percent of samples for test dataset')
    parser.add_argument('--num_classes', default=1050, type=float,
                        help='num classes to use')
    parser.add_argument('--white_list_path', default='/home/ssd_storage/experiments/vgg19_faces/top_img_cls.csv', type=str,
                        help='Path to white list file')
    parser.add_argument('--img_dir_is_cls_train', action='store_false',
                        help='Path to white list file')
    parser.add_argument('--output_dataset_dir', default='/home/ssd_storage/datasets/processed/youtube_faces', type=str,
                        help='Where to save the filtered dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    whitelist = pd.read_csv(args.white_list_path)
    whitelist = list(whitelist['class'])
    print(whitelist)
    train_size = args.train_perc
    val_size = args.val_perc
    test_size = args.test_perc
    if args.size_or_perc == 'size':
        train_size = int(args.train_perc)
        val_size = int(args.val_perc)
        test_size = int(args.test_perc)
    if args.test_perc > 0:
        phase_percentage_dict = {'train': train_size, 'val': val_size, 'test': test_size}
    else:
        phase_percentage_dict = {'train': train_size, 'val': val_size}

    cls_size = PhasePercentageProcessor(output_dataset_dir=args.output_dataset_dir, phase_percentage_dict=phase_percentage_dict)
    if args.size_or_perc == 'size':
        cls_size = PhaseSizeProcessor(output_dataset_dir=args.output_dataset_dir, phase_size_dict=phase_percentage_dict)
    # NumClassProcessor(args.num_classes, args.num_classes,args.output_dataset_dir),
    proc = MultiStageProcessor([
        ClassSizeProcessing(5, args.output_dataset_dir),
        # WhiteListProcessor(output_dataset_dir=args.output_dataset_dir, white_list=whitelist, img_dir_is_cls=args.img_dir_is_cls_train),
        # cls_size,
        PhaseSizeProcessor(output_dataset_dir=args.output_dataset_dir, phase_size_dict={'train':4, 'val':1})
    ])

    print(proc.process_dataset(raw_dataset_dir=args.source_dataset, dataset_name=args.dataset_name))
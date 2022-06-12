from glob import glob

from tqdm import tqdm
from argparse import ArgumentParser
from shutil import copyfile

from os import path


def generate_fn(file_path: str, net_name: str) -> str:
    dataset_name = path.basename(path.dirname(file_path))
    return f'{net_name}_{dataset_name}_fc7.csv'


def copy_dataset(net_dir: str, net_name: str, output_dir: str) -> None:
    rdms = glob(path.join(net_dir, '*', 'fc7.csv'))
    for rdm in tqdm(rdms):
        new_fn = generate_fn(rdm, net_name)
        output_path = path.join(output_dir, new_fn)
        copyfile(rdm, output_path)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--source_dataset', default='/home/ssd_storage/experiments/students/Noam/results_for_conference_dec21/objects_net_test/vgg16/results', type=str, help='Path to src dataset')
    parser.add_argument('--output_dataset',
                        default='/home/ssd_storage/experiments/birds/inversion/all_rdm_data',
                        type=str, help='Where to save the files to')
    parser.add_argument('--net_name',
                        default='260_objects',
                        type=str, help='name of the network')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    copy_dataset(args.source_dataset, args.net_name, args.output_dataset)
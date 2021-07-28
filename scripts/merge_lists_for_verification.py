import pandas as pd
import argparse


def merge_lists(lists_root):
    diff_path = lists_root + "_diff.txt"
    same_path = lists_root + "_same.txt"
    verification_path = lists_root + "_verification.txt"

    same_df = pd.read_csv(same_path, sep=' ', header=None, index_col=False)
    same_df['label'] = 1
    diff_df = pd.read_csv(diff_path, sep=' ', header=None, index_col=False)
    diff_df['label'] = 0

    joined_df = same_df.append(diff_df)

    joined_df.to_csv(verification_path, sep=' ', index=False, header=False)
    print(f"written to {verification_path}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs_root", type=str,
                        default="/home/administrator/experiments/familiarity/dataset/image_pairs_lists/mutualy_exclusive/pretraining")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # files = get_pairs_lists(args.pairs_dir, args.depth)
    merge_lists(args.pairs_root)
    print("Done")

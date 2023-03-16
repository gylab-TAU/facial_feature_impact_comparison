import os
import shutil
from typing import List

from tqdm import tqdm


def transfer_datapoints(dest_dataset_loc: str, source_path: str, data_points: List[str]) -> None:
    """
    Create a symlink (or copy, if symlink cannot be made) for all data_points in source_path to the new dest_dataset_loc

    :dest_dataset_loc: The destination directory, (where to move the files to)
    :source_path: The source directory containing the files
    :data_points: The absolute path of all files to copy
    """
    # pbar = tqdm(data_points)
    for point in data_points:
        dest_point = os.path.join(dest_dataset_loc, os.path.relpath(point, source_path))
        os.makedirs(os.path.dirname(dest_point), exist_ok=True)
        try:
            os.symlink(os.path.abspath(point), dest_point)
        except OSError:
            # pbar.set_description("Encountered error on point. Copying...")
            shutil.copyfile(point, dest_point)


def transfer_datapoints_to_phase2(dest_dataset_loc, phase, data_class, data_points):
    dest_dir = os.path.join(dest_dataset_loc, phase, data_class)

    os.makedirs(dest_dir)
    for source_point in data_points:
        # source_point = os.path.join(source_dataset_loc, data_class, point)
        dest_point = os.path.join(dest_dir, os.path.basename(source_point))

        try:
            os.symlink(source_point, dest_point)
        except OSError:
            shutil.copyfile(source_point, dest_point)

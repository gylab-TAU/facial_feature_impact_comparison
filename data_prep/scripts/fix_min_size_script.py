import os
import glob
import tqdm
from data_prep.util import transfer_datapoints

if __name__ == '__main__':
    print(os.getcwd())
    existing_ds_root = '/home/administrator/datasets/images_faces/faces_only'
    dest_root = '/home/administrator/datasets/processed/images_faces/faces_only {train: 300, val: any}'
    os.makedirs(dest_root, exist_ok=True)

    max_classes = 745
    min_train_datapoints = 300
    train_classes = glob.glob(os.path.join(existing_ds_root, 'train', '*'))
    class_names = [os.path.basename(cl) for cl in train_classes]
    num_classes = 0
    min_val_datapoints = 0

    for cl in tqdm.tqdm(class_names, desc='classes'):
        # Checking we have enough train
        train_cl_source = os.path.join(existing_ds_root, 'train', cl)
        train_data_points = glob.glob(os.path.join(train_cl_source, '*'))
        if len(train_data_points) >= min_train_datapoints:
            # Checking we have enough val
            val_cl_source = os.path.join(existing_ds_root, 'val', cl)
            val_data_points = glob.glob(os.path.join(val_cl_source, '*'))
            if len(val_data_points) >= min_val_datapoints:
                num_classes += 1
                # transferring class train ds
                train_dest = os.path.join(dest_root, 'train', cl)
                os.makedirs(train_dest)
                transfer_datapoints(train_dest, train_cl_source, train_data_points)

                # transferring class val ds
                val_dest = os.path.join(dest_root, 'val', cl)
                os.makedirs(val_dest)
                transfer_datapoints(val_dest, val_cl_source, val_data_points)

                if num_classes == max_classes:
                    break

    print('Moved ', num_classes, ' to ', dest_root)
    print('Done.')

import torch.utils.data
from glob import glob
import os
import random
import pandas as pd

from PIL import Image


class TripletDataset(torch.utils.data.Dataset):
    """
    A dataset for triplet loss
    Randomly sampling a positive & negative examples
    """
    def __init__(self, dir_path, transforms):
        self.transforms = transforms
        self.dir_path = dir_path
        self.classes = glob(os.path.join(dir_path, '*'))
        df = {'cls': [], 'img': []}
        for i, cl in enumerate(self.classes):
            cl_imgs = glob(os.path.join(cl, '*'))
            df['cls'] = df['cls'] + [cl] * len(cl_imgs)
            df['img'] = df['img'] + cl_imgs

        self.dataset = pd.DataFrame(df)
        pos = (self.dataset.groupby('cls').count() > 1)
        pos = pos.rename(columns={'img': 'pos'})
        self.dataset = pd.merge(self.dataset, pos, left_on='cls', right_index=True)

    def __load_img(self, img_path):
        im = Image.open(img_path)
        # fix bug from grayscale images
        # duplicate to make 3 channels
        if im.mode != 'RGB':
            im = im.convert('RGB')

        return self.transforms(im)

    def sample_pos(self, anchor_path, anchor_cls):
        same_cls_diff_img_filter = (self.dataset['cls'] == anchor_cls) & (self.dataset['img'] != anchor_path)
        pos_path = self.dataset[same_cls_diff_img_filter].sample()['img'].item()
        return pos_path

    def sample_neg(self, anchor_cls):
        diff_cls_filter = (self.dataset['cls'] != anchor_cls)
        neg_path = self.dataset[diff_cls_filter].sample()['img'].item()
        return neg_path

    def __getitem__(self, idx):
        anchor_path = self.dataset[self.dataset['pos']].iloc[idx]['img']
        anchor_cls = self.dataset[self.dataset['pos']].iloc[idx]['cls']

        pos_path = self.sample_pos(anchor_path, anchor_cls)

        neg_path = self.sample_neg(anchor_cls)

        anchor = self.__load_img(anchor_path)
        positive = self.__load_img(pos_path)
        negative = self.__load_img(neg_path)

        return anchor, positive, negative

    def __len__(self):
        return self.dataset[self.dataset['pos']].shape[0]

from torch.utils.data import DataLoader
from torch import cuda
from tqdm import tqdm
import const
import pandas as pd
import modelling.models.verification_model


class VerificationMeasurer(object):
    def __init__(self, dataset: DataLoader, dataset_name: str):
        self.__dataset = dataset
        self.__dataset_name = dataset_name

    def test_performance(self, model: modelling.models.verification_model.VerificationModel) -> pd.DataFrame:
        all_measurements = {}
        for im1, im2, key1, key2, label in tqdm(self.__dataset, desc=self.__dataset_name):
            if cuda.is_available() and const.DEBUG is False:
                im1.cuda(non_blocking=True)
                im2.cuda(non_blocking=True)
            model.verify()
            layered_verification_scores = model(im1, im2)
            for key in layered_verification_scores:
                layered_verification_scores[key] = layered_verification_scores[key]
            layered_verification_scores['im1'] = key1
            layered_verification_scores['im2'] = key2
            layered_verification_scores['label'] = label

        return pd.DataFrame(all_measurements)

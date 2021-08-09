import torch
import pandas as pd
import const
from tqdm import tqdm

class StrongestActivatingImageRetrieval(object):
    def __init__(self, dataset: torch.utils.data.DataLoader):
        self.dataset = dataset

    def compare_lists(self, model: torch.nn.Module) -> pd.DataFrame:
        top_activating_img_score = {}
        top_activating_img_file = {}
        model.train(False)
        with torch.no_grad():
            for imgs, labels, paths in tqdm(self.dataset, desc='strongest activating images search'):
                if const.DEBUG is False:
                    imgs.cuda(non_blocking=True)
                scores = model(imgs)
                for i in range(imgs.size()[0]):
                    l = int(labels[i])
                    if l not in top_activating_img_score:
                        top_activating_img_score[l] = float(scores[i][l])
                        top_activating_img_file[l] = str(paths[i])
                    if top_activating_img_score[l] < float(scores[i][l]):
                        top_activating_img_score[l] = float(scores[i][l])
                        top_activating_img_file[l] = str(paths[i])

        return pd.DataFrame({
            'score': pd.Series(top_activating_img_score),
            'file': pd.Series(top_activating_img_file),
        })

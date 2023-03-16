import torch
import pandas as pd
from tqdm import tqdm
import const


class IdentificationErrorAcquisition(object):
    def __init__(self, data_loader):
        self.__data_loader = data_loader
        self.__idx_to_class = {}
        for cl in self.__data_loader.dataset.class_to_idx:
            self.__idx_to_class[self.__data_loader.dataset.class_to_idx[cl] + 8749] = cl

    def compare_lists(self, model):
        ident_err = {'path': [], 'predicted_class_idx': [], 'predicted_class': [], 'correct_class': [], 'correct_class_rank': [], 'predicted_class_score': [], 'correct_class_score': [], 'correct_class_idx':[]}
        num_batches = len(self.__data_loader)

        # switch to modelling mode
        model.train(False)
        with torch.set_grad_enabled(False):
            data_loader_iter = iter(self.__data_loader)
            pbar = tqdm(range(num_batches), desc='Identification errors')
            for i in pbar:
                (images, target, path) = next(data_loader_iter)

                output, target = self.__per_batch(model, images, target)
                output = output.cpu()
                target = target.cpu()

                target = target.unsqueeze(1)
                scores, preds = torch.max(output, 1)
                preds = preds.cpu()
                if torch.sum(preds == target.data).item() < 1:
                    print(preds)
                    print(target[0].data.item())
                    print(output[0].size())
                    ident_err['path'].append(path)
                    ident_err['predicted_class_idx'].append(preds[0].data.item())
                    if preds[0].data.item() >= 8749:
                        ident_err['predicted_class'].append(self.__idx_to_class[preds[0].data.item()])
                    else:
                        ident_err['predicted_class'].append(None)
                    ident_err['correct_class'].append(self.__idx_to_class[target[0].data.item()])
                    ident_err['correct_class_idx'].append(target[0].data.item())
                    ident_err['correct_class_rank'].append(torch.sum(output[0] > output[0][target[0].data.item()]).item())
                    ident_err['predicted_class_score'].append(output[0][preds.data.item()].data.item())
                    ident_err['correct_class_score'].append(output[0][target[0].data.item()].data.item())

            df = pd.DataFrame(ident_err)
            print(df)
            return df

    def __per_batch(self, model, images, target):
        if torch.cuda.is_available() and const.DEBUG is False:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        if type(output) is not torch.Tensor:
            output = output[0]

        return output, target

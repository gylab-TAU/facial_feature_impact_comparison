import torch
import pandas as pd
from tqdm import tqdm


class ActivationAcquisition(object):
    def __init__(self, data_loader, whitelist: list = [], num_classes:int = 1000):
        self.__data_loader = data_loader
        self.__whitelist = [data_loader.dataset.class_to_idx[cl] for cl in whitelist]
        self.__num_classes = num_classes

    def compare_lists(self, model):
        activations_df = None
        num_batches = len(self.__data_loader)

        # switch to modelling mode
        model.train(False)
        with torch.set_grad_enabled(False):
            data_loader_iter = iter(self.__data_loader)
            pbar = tqdm(range(num_batches), desc='Activations')
            for i in pbar:
                (images, target) = next(data_loader_iter)

                output, target = self.__per_batch(model, images, target)
                if target.size()[0] != 0:
                    target = target.unsqueeze(1)
                    # print(output.size(), target.size())
                    joined = torch.cat((output, target), 1)
                    df = pd.DataFrame(joined).astype("float")
                    if activations_df is None:
                        activations_df = df
                    else:
                        activations_df = activations_df.append(df)
            return activations_df

    def __per_batch(self, model, images, target):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        if type(output) is not torch.Tensor:
            output = output[0]

        good_target = torch.Tensor([t in self.__whitelist for t in target])
        target = target[good_target.nonzero().squeeze()]
        output = output[good_target.nonzero().squeeze()]

        return output, target

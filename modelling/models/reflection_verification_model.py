from typing import Optional, Any
from torch import nn, Tensor
from modelling.models.verification_model import VerificationModel
from representation.acquisition.representation_extraction import RepresentationExtractor
from representation.acquisition.representation_save_hook import FileSystemHook


class ReflectionVerificationModel(VerificationModel):
    def __init__(self,
                 inner: nn.Module,
                 reps_cache_path: str,
                 verification_score_calc: nn.Module = None,
                 layers_extractor: Optional[object] = None):
        """
        A model with a verification mode, using reflection to save representations to local FS and loading them back
        :param inner: The actual model to use
        :param reps_cache_path: Where to save the representations (temporarily)
        :param verification_score_calc: How to calculate verification score_calc
        :param layers_extractor: Class extracting which layers to use in order to extract representations
        """
        super(self, VerificationModel).__init__()
        self.__inner = inner
        self.__layers_extractor = layers_extractor
        self.__reps_cache_path = reps_cache_path
        self.__verification_score_calc = verification_score_calc

    def set_verification_model(self, layers_extractor) -> None:
        self.__layers_extractor = layers_extractor

    def forward(self, t1: Tensor, t2: Optional[Tensor] = None) -> Any:
        if not self.verify:
            return self.__inner(t1)
        else:
            re = RepresentationExtractor(self.__inner,
                                         self.__layers_extractor(self.__inner),
                                         FileSystemHook(self.__layers_extractor(self.__inner), self.__reps_cache_path,
                                                        delete_after_load=True))

            rep1 = re.get_layers_representation(t1, 'key1')
            rep2 = None
            if t2 is not None:
                rep2 = re.get_layers_representation(t2, 'key2')

            del re

            # if the model should work on 2 tensors:
            if rep2 is not None:
                # if the model should produce verifications
                if self.__verification_score_calc is not None:
                    verification_output = {}
                    for key in rep1:
                        verification_output[key] = self.__verification_score_calc(rep1[key], rep2[key])
                    return verification_output
                # if the model should produce representations
                else:
                    return rep1, rep2
            # if the model should produce single tensor representations
            else:
                return rep1

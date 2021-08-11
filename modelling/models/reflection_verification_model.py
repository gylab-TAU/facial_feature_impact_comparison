from typing import Optional, Any
from torch import nn, Tensor
from modelling.models.verification_model import VerificationModel
from representation.acquisition.representation_extraction import RepresentationExtractor
from representation.acquisition.representation_save_hook import FileSystemHook


class ReflectionVerificationModel(VerificationModel):
    def __init__(self,
                 inner: nn.Module,
                 verification_score_calc: nn.Module,
                 reps_cache_path: str,
                 layers_extractor: Optional[object] = None):
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
            rep2 = re.get_layers_representation(t2, 'key2')

            del re

            verification_output = {}
            for key in rep1:
                verification_output[key] = self.__verification_score_calc(rep1[key], rep2[key])
            return verification_output

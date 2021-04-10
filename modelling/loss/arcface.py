from typing import Optional
import torch


class ArcFace(torch.nn.CrossEntropyLoss):
    """
    Implementaion of ArcFace loss adapted from "https://github.com/deepinsight/insightface/blob/fec2bf0b28c96aba9a4eacbae32fabdebd6bee94/recognition/arcface_torch/losses.py"
    article: "https://arxiv.org/pdf/1801.07698.pdf"
    """
    def __init__(self, scale_factor=64.0, margin=0.5, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        self.scale_factor = scale_factor
        self.margin = margin
        super(ArcFace, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param input: Arccos of cosine similarity between weight matrix and features output.
        :param target: correct labels of each angle data
        :return: CrossEntropyLoss of added margin cosine similarity (rescaled)
        """
        index = torch.where(target != -1)[0]
        m_hot = torch.zeros(index.size()[0], input.size()[1], device=input.device)
        m_hot.scatter_(1, target[index, None], self.m)
        input.acos_()
        input[index] += m_hot
        input.cos_().mul_(self.s)
        return super(ArcFace, self)(input, target)

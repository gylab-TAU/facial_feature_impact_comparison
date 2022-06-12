from torch import Tensor, nn


class CosineDistance(nn.CosineSimilarity):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineDistance, self).__init__(dim, eps)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return 1 - super(CosineDistance,self).forward(x1, x2)
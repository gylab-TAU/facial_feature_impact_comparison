from typing import Any, Dict, List, Union
import torch
import torchvision
from torchvision.models import vgg

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

arch_to_cfg = {
    'vgg11': 'A',
    'vgg11_bn': 'A',
    'vgg13': 'B',
    'vgg13_bn': 'B',
    'vgg16': 'D',
    'vgg16_bn': 'D',
    'vgg19': 'E',
    'vgg19_bn': 'E'
}

batch_norm = {
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn'
}


class ArcVGG(torchvision.models.VGG):
    def __init__(
            self,
            arch: str, num_classes: int = 1000, **kwargs: Any) -> None:
        super(ArcVGG, self).__init__(vgg.make_layers(cfgs[arch_to_cfg [arch]], batch_norm=arch in batch_norm), num_classes, **kwargs)
        self.classifier[-1] = torch.nn.Linear(4096, num_classes, bias=False)

    def cosine_sim_per_line(self, x1: torch.Tensor, x2: torch.Tensor):
        x1_norm = x1 / x1.norm(dim=1)[:, None]
        x2_norm = x2 / x2.norm(dim=1)[:, None]
        res = torch.mm(x1_norm, x2_norm.transpose(0, 1))
        return res

    def forward(self, x: torch.Tensor):
        if not self.training:
            return super(ArcVGG, self)(x)
        elif self.training:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            for i in len(self.classifier) - 1:
                x = self.classifier[i](x)
            x = self.cosine_sim_per_line(x, self.classifer[-1].weights)
            x = torch.arccos(x)
            return x

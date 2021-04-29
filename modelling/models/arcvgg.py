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

    def forward(self, x: torch.Tensor):
        # if not self.training:
        #     return super(ArcVGG, self).forward(x)
        # elif self.training:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.classifier) - 3):
            x = self.classifier[i](x)
        x = x / x.norm(dim=1)[:, None]
        x = self.classifier[-3](x) # ReLU
        x = self.classifier[-2](x) # Dropout

        x = self.classifier[-1](x) # FC8
        x = x / self.classifier[-1].weight.norm(dim=1)
        return x

from torch import nn, Tensor


class OverlaidClassifier(nn.Module):
    def __init__(self, projector: nn.Module, emb_dim: int, num_classes: int = 1000):
        super(OverlaidClassifier, self).__init__()
        self.projector = projector
        self.relu = nn.ReLU()
        self.num_classes = num_classes
        self.classifier = nn.Linear(emb_dim, num_classes)
        print(self.classifier)

    def forward(self, x: Tensor):
        projection = self.projector(x)
        non_neg_projection = self.relu(projection)
        scores = self.classifier(non_neg_projection)
        return scores
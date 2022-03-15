import torchvision
import torch
import const
from modelling.models.overlaid_classifier import OverlaidClassifier


def append_classes(model: torch.nn.modules.Module, num_new_classes):
    """
    Appending classes to the existing last layer (must be FC)
    If type(last layer) != FC: raises assertion error.
    :param model: the model to append classes to
    :param num_new_classes: the number of new classes to append
    :return: the updated model
    """
    last_layer = None
    for layer in model.modules():
        last_layer = layer
    assert type(last_layer) == torch.nn.modules.linear.Linear
    addition = torch.nn.modules.linear.Linear(last_layer.in_features, num_new_classes, last_layer.bias is not None)
    if torch.has_cuda and const.DEBUG is False:
        addition.cuda()
    # weights:

    joined_classes_weight = torch.cat((last_layer.weight.data, addition.weight.data))

    #bias:
    # device = last_layer.bias.device
    #
    # if torch.has_cuda:
    #     new_classes_bias = torch.cuda.FloatTensor(num_new_classes)
    # else:
    #     new_classes_bias = torch.FloatTensor(num_new_classes)

    joined_classes_bias = torch.cat((last_layer.bias.data, addition.bias.data))

    last_layer.weight = torch.nn.Parameter(joined_classes_weight)
    last_layer.bias = torch.nn.Parameter(joined_classes_bias)

    return model


def replace_classes(model: torch.nn.modules.Module, num_new_classes):
    if type(model) == torchvision.models.resnet.ResNet:
        last_layer = model.fc
        assert type(last_layer) == torch.nn.modules.linear.Linear
        replacement = torch.nn.modules.linear.Linear(last_layer.in_features, num_new_classes,
                                                     last_layer.bias is not None)
        replacement.train(True)
        replacement.requires_grad = True

        if torch.has_cuda and const.DEBUG is False:
            replacement.cuda()
        model.fc = replacement
        return model
    if type(model) == torchvision.models.VGG:
        last_layer = model.classifier[-1]
        assert type(last_layer) == torch.nn.modules.linear.Linear
        replacement = torch.nn.modules.linear.Linear(last_layer.in_features, num_new_classes, last_layer.bias is not None)
        replacement.train(True)
        replacement.requires_grad = True
        if torch.has_cuda and const.DEBUG is False:
            replacement.cuda()
        model.classifier[-1] = replacement
        return model


def overlay_classes(model: torch.nn.modules.Module, emb_dim: int, num_new_classes: int):
    model = OverlaidClassifier(model, emb_dim, num_new_classes)
    if torch.cuda.is_available():
        model.cuda()
    return model
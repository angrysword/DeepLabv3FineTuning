""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch import nn

def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True,num_classes=21)

    #model.classifier = DeepLabHead(2048, outputchannels)
    model.backbone.conv1=nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(3, 3), bias=False)
    model.classifier = DeepLabHead(2048, 1)
    # Set the model in training mode
    model.train()
    return model

import torch.nn as nn
from torchvision import models


class StyleTransfer(nn.Module):
    """
    Forward propagate an Image through VGG19 and
    return the feature maps for certain layers.
    """

    def __init__(self):
        super().__init__()

        # layers for content and style representation
        self.layers = {
            '1': 'relu1_1',
            '6': 'relu2_1',
            '11': 'relu3_1',
            '20': 'relu4_1',
            '22': 'relu4_2',  # content representation
            '29': 'relu5_1'
        }
        # load the vgg19 pretrained model and freeze the parameters
        self.model = models.vgg19(pretrained=True).features[:30]
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, image):
        features = {}
        x = image

        # model._modules is a dictionary with each module in the model
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x

        return features

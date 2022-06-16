import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

import timm


class LINEAR_CLASSIFIER(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, in_size, out_size):
        super().__init__()
        self.net = nn.Linear(in_size, out_size)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x



class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class LINEAR_CLASSIFIER(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, in_size, out_size):
        super().__init__()
        self.net = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.net(x)


MODELS_SUPORTED = [
    "efficientnet_b7",
    "face_resnet",
    "vision_transformer",
    "xception",
    "mesonet",
]


def instantiate_model(model_name, pretrained=True, embedder=True, embedding_size=128):

    # Set trunk model and replace the softmax layer with an identity function

    # model variable part

    if model_name == "efficientnet_b7":
        trunk = timm.create_model("tf_efficientnet_b7_ns", pretrained=pretrained)
        trunk_output_size = trunk.classifier.in_features
        trunk.classifier = Identity()

    elif model_name == "face_resnet":
        if pretrained == True:

            trunk = InceptionResnetV1(pretrained="vggface2")
        else:
            trunk = InceptionResnetV1()

        trunk_output_size = trunk.last_bn.in_features
        trunk.classifier = Identity()

    elif model_name == "vision_transformer":
        trunk = timm.create_model("vit_base_patch8_224", pretrained=pretrained)

        trunk_output_size = trunk.head.in_features
        trunk.head = Identity()

    elif model_name == "xception":
        trunk = timm.create_model("xception", pretrained=pretrained)

        trunk_output_size = trunk.fc.in_features
        trunk.fc = Identity()

    else:
        raise ValueError(f"{model_name} must be one of {MODELS_SUPORTED}")
    # common for models

    if embedder == True:

        # Set embedder model. This takes in the output of the trunk and outputs n dimensional embeddings

        MLP([trunk_output_size, embedding_size])

        classifier = LINEAR_CLASSIFIER(embedding_size, 1)

        return trunk, embedder, classifier

    else:
        classifier = LINEAR_CLASSIFIER(trunk_output_size, 1)

        return trunk, classifier

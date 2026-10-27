"""The three models compared in the paper.

Every model's forward pass returns the same named tuple so that one training
loop and one evaluation routine can serve all of them:

    logits: (n, 4) genre scores, or None for the unsupervised autoencoder
    recon:  (n, 500) reconstruction, or None for the plain classifier
    z:      (n, 64) bottleneck encoding, or None for the plain classifier
"""
from typing import NamedTuple, Optional

import torch
from torch import nn

from . import config as C


class Output(NamedTuple):
    logits: Optional[torch.Tensor]
    recon: Optional[torch.Tensor]
    z: Optional[torch.Tensor]


def mlp(sizes, dropout: float, final_activation: bool) -> nn.Sequential:
    """Fully connected stack with tanh units and dropout between layers."""
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        last = i == len(sizes) - 2
        if not last or final_activation:
            layers.append(nn.Tanh())
        if not last and dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TwoLayerNet(nn.Module):
    """Baseline classifier: 500 -> 128 (tanh) -> 4."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.net = mlp((C.INPUT_DIM, C.BASELINE_HIDDEN, len(C.GENRES)), dropout, final_activation=False)

    def forward(self, x):
        return Output(self.net(x), None, None)


class Autoencoder(nn.Module):
    """Vanilla autoencoder: 500 -> 256 -> 192 -> 128 -> 64 and back."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        enc = (C.INPUT_DIM, *C.ENCODER_SIZES, C.ENCODING_DIM)
        self.encoder = mlp(enc, dropout, final_activation=True)
        self.decoder = mlp(enc[::-1], dropout, final_activation=False)

    def forward(self, x):
        z = self.encoder(x)
        return Output(None, self.decoder(z), z)


class DeepSoftmaxAutoencoder(Autoencoder):
    """Autoencoder whose 64-dim encoding also feeds a 64 -> 32 -> 16 -> 4 classifier."""

    def __init__(self, dropout: float = 0.1):
        super().__init__(dropout)
        sizes = (C.ENCODING_DIM, *C.CLASSIFIER_SIZES, len(C.GENRES))
        self.classifier = mlp(sizes, dropout, final_activation=False)

    def forward(self, x):
        z = self.encoder(x)
        return Output(self.classifier(z), self.decoder(z), z)


MODELS = {
    "two_layer": TwoLayerNet,
    "autoencoder": Autoencoder,
    "dsae": DeepSoftmaxAutoencoder,
}
# Names of the training runs. A run is an architecture, optionally trained with
# random-crop augmentation (config.AUGMENT_SUFFIX).
DISPLAY_NAMES = {
    "two_layer": "Two Layer Neural Network",
    "two_layer_aug": "Two Layer Neural Network + random crops",
    "autoencoder": "Vanilla Autoencoder",
    "dsae": "Deep Softmax Autoencoder",
    "dsae_aug": "Deep Softmax Autoencoder + random crops",
}
SHORT_NAMES = {       # for figure titles and axis labels, where space is tight
    "two_layer": "Two-layer network",
    "two_layer_aug": "Two-layer network + random crops",
    "autoencoder": "Vanilla autoencoder",
    "dsae": "DSAE",
    "dsae_aug": "DSAE + random crops",
}


def build_model(name: str, dropout: float = 0.1) -> nn.Module:
    model = MODELS[name](dropout=0.0 if name == "two_layer" else dropout)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

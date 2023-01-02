import torch
import torch.nn as nn


class MnistClassifierMlp(nn.Module):

    def __init__(self, features_dim: int):
        super().__init__()
        self.__features_dim = features_dim
        self.linear = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        self.last = nn.Sequential(nn.Linear(features_dim, 4))  # Subject to be replaced dependent on task

    def features_dim(self):
        return self.__features_dim

    def features(self, x):
        x = self.linear(torch.flatten(x, 1))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

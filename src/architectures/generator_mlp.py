import torch
import torch.nn as nn


class GeneratorMlp(nn.Module):
    def __init__(self, noise_dim: int, features_dim: int):
        super().__init__()
        self.__noise_dim = noise_dim
        self.__sequential_blocks = [
            nn.Linear(noise_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def get_noise_dim(self):
        return self.__noise_dim

    def forward(self, input_latent: torch.Tensor):
        decoded_images = self.main(input_latent)
        return decoded_images

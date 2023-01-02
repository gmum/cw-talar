import torch
from architectures.mnist_classifier_mlp import MnistClassifierMlp
from torch.utils.data import DataLoader


def get_activations(network: MnistClassifierMlp, dataloader: DataLoader, all: bool = False):
    features = list()
    for batch, _ in dataloader:
        batch = batch.cpu()
        result = network.features(batch)
        features.append(result)

    all_activations = torch.cat(features, 0)
    print(torch.count_nonzero(all_activations, dim=0))
    # print(torch.max(all_actv, dim=0))
    if all:
        return all_activations
    return all_activations[0:1000]

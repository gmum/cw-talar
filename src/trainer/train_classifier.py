import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics.accuracy import accuracy


def train_classifier(network: nn.Module, dataloader: DataLoader, epochs_count: int) -> None:
    optimizer = torch.optim.Adam(network.parameters())
    criterion_fn = nn.CrossEntropyLoss().cpu()

    for _ in range(epochs_count):
        accuracies = list()
        for batch, target in dataloader:
            target = target.cpu()
            result = network(batch.cpu())
            cost = criterion_fn(result, target)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            accuracy_now = accuracy(result, target)
            # print(accuracy_now)
            accuracies.append(torch.FloatTensor([accuracy_now]))

        all_accuracies = torch.cat(accuracies, 0)
        print('ACCURACY: ', all_accuracies.mean())

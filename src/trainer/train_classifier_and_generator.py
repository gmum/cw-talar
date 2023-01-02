import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from architectures.generator_mlp import GeneratorMlp
from architectures.mnist_classifier_mlp import MnistClassifierMlp
from cw_torch.metric import cw
from cw_torch.gamma import silverman_rule_of_thumb_sample
from metrics.accuracy import accuracy
from tqdm import tqdm

def train_classifier_with_generator(network: MnistClassifierMlp, generator: GeneratorMlp, dataloader: DataLoader, epochs_count: int = 4):
    print('Train classifier with generator')

    optimizer = torch.optim.Adam(network.parameters())
    criterion_fn = nn.CrossEntropyLoss().cpu()

    for _ in range(epochs_count):
        accuracies = list()
        for batch, target in dataloader:
            batch = batch.cpu()
            target = target.cpu()
            features = network.features(batch)
            result = network.logits(features)
            noise = torch.randn(batch.size(0), generator.get_noise_dim()).cpu()
            generated_features = generator(noise)
            gamma = silverman_rule_of_thumb_sample(torch.cat((features, generated_features), 0))
            cw_cost = cw(features, generated_features, gamma)
            cost = criterion_fn(result, target) + 10000*cw_cost
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            accuracy_now = accuracy(result, target)
            accuracies.append(torch.FloatTensor([accuracy_now]))
        all_accuracies = torch.cat(accuracies, 0)
        tqdm()
        print('ACCURACY: ', all_accuracies.mean())

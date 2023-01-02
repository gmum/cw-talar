import torch
import numpy as np
from architectures.generator_mlp import GeneratorMlp
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from cw_torch.metric import cw
from cw_torch.gamma import silverman_rule_of_thumb_sample


def train_generator(generator: GeneratorMlp, activations: np.ndarray, generator_epochs: int):
    print('Train generator')
    dataset = TensorDataset(torch.FloatTensor(activations))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                            num_workers=0, drop_last=True, pin_memory=False)

    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    for _ in range(generator_epochs):
        cw_distances = list()
        for batch in dataloader:
            batch = batch[0].cpu()
            noise = torch.randn(batch.size(0), generator.get_noise_dim()).cpu()
            result = generator(noise)
            gamma = silverman_rule_of_thumb_sample(torch.cat((result, batch), 0))
            cost = cw(result, batch, gamma)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            cw_distances.append(torch.FloatTensor([cost]))
        all_accuracies = torch.cat(cw_distances, 0)
        print('CW_Distance: ', all_accuracies.mean())

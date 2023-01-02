
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def create_mnist_dataloader_for_classes(mnist_dataset: MNIST, index1: int, index2: int) -> DataLoader:
    idx_one = mnist_dataset.targets == index1
    idx_two = mnist_dataset.targets == index2
    mnist_dataset.targets = mnist_dataset.targets[idx_one | idx_two]
    mnist_dataset.data = mnist_dataset.data[idx_one | idx_two]

    return DataLoader(mnist_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)


def create_mnist_dataloader_for_class(mnist_dataset: MNIST, index: int) -> DataLoader:
    idx_one = mnist_dataset.targets == index
    mnist_dataset.targets = mnist_dataset.targets[idx_one]
    mnist_dataset.data = mnist_dataset.data[idx_one]

    return DataLoader(mnist_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)

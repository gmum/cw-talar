from torchvision import transforms
from torchvision.datasets import MNIST


def get_mnist_dataset(dataroot: str, train: bool) -> MNIST:
    dataset_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return MNIST(root=dataroot,
                 train=train,
                 download=True,
                 transform=dataset_transforms)

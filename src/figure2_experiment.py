import torch
from architectures.generator_mlp import GeneratorMlp
from architectures.mnist_classifier_mlp import MnistClassifierMlp
from data.mnist_dataloader import create_mnist_dataloader_for_class, create_mnist_dataloader_for_classes
from data.mnist_dataset_factory import get_mnist_dataset
from trainer.train_classifier import train_classifier
from trainer.train_generator import train_generator
from trainer.train_classifier_and_generator import train_classifier_with_generator
from utilities.get_activations import get_activations
from utilities.display_plot import display_plot
import copy
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_root = '../data/'
    generator_noise_dim = 8
    features_dim = 4
    classifier_epochs = 10
    generator_epochs = 400

    network = MnistClassifierMlp(features_dim).cpu()
    generator = GeneratorMlp(generator_noise_dim, features_dim).cpu()

    mnist_dataloader_01 = create_mnist_dataloader_for_classes(get_mnist_dataset(data_root, True), 0, 1)
    mnist_dataloader_0 = create_mnist_dataloader_for_class(get_mnist_dataset(data_root, True), 0)
    mnist_dataloader_1 = create_mnist_dataloader_for_class(get_mnist_dataset(data_root, True), 1)
    mnist_dataloader_23 = create_mnist_dataloader_for_classes(get_mnist_dataset(data_root, True), 2, 3)
    mnist_dataloader_2 = create_mnist_dataloader_for_class(get_mnist_dataset(data_root, True), 2)
    mnist_dataloader_3 = create_mnist_dataloader_for_class(get_mnist_dataset(data_root, True), 3)

    train_classifier(network, mnist_dataloader_01, classifier_epochs)
    copied_network = copy.deepcopy(network)

    activations_ones = get_activations(network, mnist_dataloader_0).detach().cpu().numpy()
    activations_twos = get_activations(network, mnist_dataloader_1).detach().cpu().numpy()

    train_generator(generator, np.concatenate([activations_ones, activations_twos], axis=0), generator_epochs)
    generator_activations = generator(torch.randn(1000, generator_noise_dim).cpu()).detach().cpu().numpy()
    train_classifier_with_generator(network, generator, mnist_dataloader_23, classifier_epochs)

    activations_threes = get_activations(network, mnist_dataloader_2).detach().cpu().numpy()
    activations_fours = get_activations(network, mnist_dataloader_3).detach().cpu().numpy()

    activations_ones_after_second_task = get_activations(network, mnist_dataloader_0).detach().cpu().numpy()
    activations_twos_second_task = get_activations(network, mnist_dataloader_1).detach().cpu().numpy()

    train_classifier(copied_network, mnist_dataloader_23, classifier_epochs)

    copy_activations_threes = get_activations(copied_network, mnist_dataloader_2).detach().cpu().numpy()
    copy_activations_fours = get_activations(copied_network, mnist_dataloader_3).detach().cpu().numpy()
    copy_activations_ones_after_second_task = get_activations(copied_network, mnist_dataloader_0).detach().cpu().numpy()
    copy_activations_twos_second_task = get_activations(copied_network, mnist_dataloader_1).detach().cpu().numpy()

    all_data = np.concatenate((generator_activations, activations_ones, activations_twos, activations_threes, activations_fours,
                               activations_ones_after_second_task, activations_twos_second_task, copy_activations_threes,
                               copy_activations_fours, copy_activations_ones_after_second_task, copy_activations_twos_second_task), 0)

    mean = np.mean(all_data, axis=0, keepdims=True)
    stddev = np.std(all_data, axis=0, keepdims=True) + 1e-3
    print(stddev)
    print(mean.shape, stddev.shape, all_data.shape)

    all_data = (all_data - mean) / stddev

    pca = PCA(n_components=2)
    pca = pca.fit(all_data)

    all_data = pca.transform(all_data)

    activations_ones_transformed = pca.transform((activations_ones - mean) / stddev)
    activations_twos_transformed = pca.transform((activations_twos - mean) / stddev)

    activations_threes_transformed = pca.transform((activations_threes - mean)/stddev)
    activations_fours_transformed = pca.transform((activations_fours - mean)/stddev)
    activations_ones_after_second_task_transformed = pca.transform((activations_ones_after_second_task - mean)/stddev)
    activations_twos_second_task_transformed = pca.transform((activations_twos_second_task - mean) / stddev)

    plt.xlim(all_data[:, 0].min() * 1.1, all_data[:, 0].max() * 1.1)
    plt.ylim(all_data[:, 1].min() * 1.1, all_data[:, 1].max() * 1.1)

    display_plot([activations_ones_transformed, activations_twos_transformed], 'First task')
    # display_plot([generator_activations], )
    display_plot([activations_threes_transformed, activations_fours_transformed, activations_ones_after_second_task_transformed,
                  activations_twos_second_task_transformed], 'Second task (CW-TaLaR)')

    copy_activations_threes_transformed = pca.transform((copy_activations_threes - mean)/stddev)
    copy_activations_fours_transformed = pca.transform((copy_activations_fours - mean)/stddev)
    copy_activations_ones_after_second_task_transformed = pca.transform((copy_activations_ones_after_second_task - mean)/stddev)
    copy_activations_twos_second_task_transformed = pca.transform((copy_activations_twos_second_task - mean) / stddev)

    plt.xlim(all_data[:, 0].min() * 1.1, all_data[:, 0].max() * 1.1)
    plt.ylim(all_data[:, 1].min() * 1.1, all_data[:, 1].max() * 1.1)

    display_plot([copy_activations_threes_transformed, copy_activations_fours_transformed, copy_activations_ones_after_second_task_transformed,
                  copy_activations_twos_second_task_transformed], 'Second task (no regularization)')

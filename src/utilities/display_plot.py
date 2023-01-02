import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def convert_to_df(arr, cat):
    return pd.DataFrame({'x': arr[:, 0], 'y': arr[:, 1], 'cat': cat})


def display_plot(activations_list, title: str):
    sns.set_theme(style="whitegrid")
    df = pd.concat([convert_to_df(x, i // 2) for (i, x) in enumerate(activations_list)])
    ax = sns.scatterplot(data=df, x="x", y="y", alpha=0.5, hue='cat', legend=False, palette='deep')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    plt.title(title)
    plt.show()

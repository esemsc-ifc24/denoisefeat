import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(data, title="Frequency Heatmap"):
    sns.heatmap(data, cmap="coolwarm")
    plt.title(title)
    plt.show()

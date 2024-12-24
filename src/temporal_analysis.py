import matplotlib.pyplot as plt


def plot_feature_evolution(features, steps):
    plt.plot(steps, features)
    plt.xlabel("Denoising Steps")
    plt.ylabel("Feature Metric")
    plt.title("Feature Evolution Over Time")
    plt.show()

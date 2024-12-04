import matplotlib.pyplot as plt
import torch

def visualize_features(features, timestep):
    for layer, feature_map in features.items():
        feature_map = feature_map.detach().cpu()
        plt.imshow(feature_map[0].mean(0), cmap='viridis')
        plt.title(f"Timestep {timestep}, Layer {layer}")
        plt.show()

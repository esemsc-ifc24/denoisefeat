import cv2
import numpy as np


def edge_density(image):
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges) / edges.size


def texture_metric(wavelet_coeffs):
    # Example metric based on wavelet coefficients
    return np.mean(wavelet_coeffs)


def global_shape_metric(fourier_coeffs):
    # Measure low-frequency energy
    return np.sum(fourier_coeffs[:10, :10])

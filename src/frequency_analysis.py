import numpy as np
from scipy.fft import fft2, ifft2
import pywt  # Wavelet transforms


def perform_fourier(image):
    # Apply Fourier Transform
    return np.abs(fft2(image))


def perform_wavelet(image):
    # Apply Wavelet Transform
    coeffs = pywt.dwt2(image, 'haar')
    return coeffs  # Approximation and detail coefficients

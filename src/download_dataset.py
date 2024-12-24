import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Save to HPC's storage
data_path = "/data"

# Download CIFAR-10 as an example
datasets.CIFAR10(root=data_path, download=True,
                 transform=transforms.ToTensor())

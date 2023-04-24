import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from stacked_mnist import StackedMNISTData, DataMode


class VAE(nn.Module):

    def __init__(self, n_channels, criterion, latent_dim):
        super().__init__()
        self.criterion = criterion #nn.MSELoss()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        
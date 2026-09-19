import torch
from model import VAE
from data import GetFaceDataLoader
# Params

epochs = 5
kernel_size = 4
hidden_layer = 32

data_loader = GetFaceDataLoader()



vae_model = VAE(32)
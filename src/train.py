import torch
from utils import KL
from model import VAE
from data import GetFaceDataLoader
from torchinfo import summary
import matplotlib.pyplot as plt


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Params

epochs = 2
kernel_size = 16
hidden_layer = 32
latent_dim = 128
batch_size = 32


# Loading Data

data_loader = GetFaceDataLoader(batch_size=batch_size)
# [32, 3, 128, 128]



# Model

vae_model = VAE(3, 128, hidden_dim=hidden_layer, kernel_size=kernel_size, latent_dim=latent_dim)

optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

loss_func = torch.nn.MSELoss()

print(vae_model)
summary(vae_model, input_size=(batch_size, 3, 128, 128))

print("Starting training...")

for epoch in range(epochs):
    batch_index = 0
    for batch in data_loader:
        optimizer.zero_grad()
        recon_batch, _mean, _logvar = vae_model(batch)
        loss = loss_func(recon_batch, batch) + KL(_mean, _logvar)
        loss.backward()
        optimizer.step()

        batch_index+=1
        print(f"Batch {batch_size+1}/{batch_index} completed. Loss: {loss}")

    print(f"Epoch {epoch+1}/{epochs} completed. Loss: {loss.item()}")






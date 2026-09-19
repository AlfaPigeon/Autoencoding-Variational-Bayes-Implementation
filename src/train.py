import torch
from utils import KL
from model import VAE
from data import GetFaceDataLoader
# Params

epochs = 5
kernel_size = 32
hidden_layer = 32
latent_dim = 32
batch_size = 32


data_loader = GetFaceDataLoader(batch_size=batch_size)


vae_model = VAE(128, kernel_size=kernel_size, latent_dim=latent_dim, hidden_layer=hidden_layer)

optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

loss_func = torch.nn.MSELoss()


for epoch in range(epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        recon_batch, _mean, _logvar = vae_model(batch)
        loss = loss_func(recon_batch, batch) + KL(_mean, _logvar)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} completed. Loss: {loss.item()}")





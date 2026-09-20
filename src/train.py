import torch
from utils import KL
from model import VAE
from data import GetFaceDataLoader
from torchinfo import summary

# Params

epochs = 5
kernel_size = 32
hidden_layer = 32
latent_dim = 128
batch_size = 32


#data_loader = GetFaceDataLoader(batch_size=batch_size)


# [32, 3, 128, 128]


vae_model = VAE(3, 128, hidden_dim=hidden_layer, kernel_size=kernel_size, latent_dim=latent_dim)





optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

loss_func = torch.nn.MSELoss()

print(vae_model)
summary(vae_model, input_size=(batch_size, 3, 128, 128))

exit()

print("Starting training...")

for epoch in range(epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        recon_batch, _mean, _logvar = vae_model(batch)
        loss = loss_func(recon_batch, batch) + KL(_mean, _logvar)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} completed. Loss: {loss.item()}")





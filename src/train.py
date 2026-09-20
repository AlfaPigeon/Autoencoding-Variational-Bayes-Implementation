import torch
from utils import KL, PlotReconstructions
from model import VAE
from data import GetFaceDataLoader, GetFaceTrainTestDataLoaders
from torchinfo import summary


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Params

epochs = 10
kernel_size = 32
hidden_layer = 128
latent_dim = 256
batch_size = 32
kl_weight = 0.0005

# Loading Data

train_loader, test_loader = GetFaceTrainTestDataLoaders(train_size=batch_size*50, test_size=batch_size*10, batch_size=batch_size)

# Train set, Test set split



# [32, 3, 128, 128]



# Model

vae_model = VAE(3, 128, hidden_dim=hidden_layer, kernel_size=kernel_size, latent_dim=latent_dim)

optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-5)

loss_func = torch.nn.MSELoss()

print(vae_model)
summary(vae_model, input_size=(batch_size, 3, 128, 128))

print("Starting training...")

for epoch in range(epochs):
    batch_index = 0
    for batch in train_loader:
        optimizer.zero_grad()
        recon_batch, _mean, _logvar = vae_model(batch)
        loss = loss_func(recon_batch, batch) + kl_weight *KL(_mean, _logvar)
        loss.backward()
        optimizer.step()

        batch_index+=1

        print(f"Batch {batch_index+1}/{len(train_loader)} completed. Loss: {loss}")

    # Evaluate on test set
    with torch.no_grad():
        test_loss = 0
        for batch in test_loader:
            recon_batch, _mean, _logvar = vae_model(batch)
            test_loss += loss_func(recon_batch, batch) + kl_weight * KL(_mean, _logvar)
        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1}/{epochs} test loss: {test_loss.item()}")

    # Plot some reconstructed images from the test set
    with torch.no_grad():
        for batch in test_loader:
            recon_batch, _, _ = vae_model(batch)
            PlotReconstructions(batch, recon_batch)
            break

    print(f"Epoch {epoch+1}/{epochs} completed. Loss: {loss.item()}")






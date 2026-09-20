import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, input_dim1=3, input_dim2=128, hidden_dim=32, latent_dim=128, kernel_size=4):
        super(VAE, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        # Shape Calculatios===
        self.conv_out_size = (input_dim2 - kernel_size) // 2 + 1
        self.flattened_dim = self.conv_out_size * self.conv_out_size * hidden_dim 
        self.hidden_dim = hidden_dim
        # ====================

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim1, hidden_dim, kernel_size=kernel_size, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space
        self.fc_mean = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_dim),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim, self.conv_out_size, self.conv_out_size)),
            nn.ConvTranspose2d(hidden_dim, input_dim1, kernel_size=kernel_size, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, _mean, _logvar):
        std = torch.exp(0.5*_logvar)
        e = torch.randn_like(std)
        return _mean + e*std

    def decode(self, z):
        recon = self.decoder(z)
        return recon
        
    def encode(self, x):
        _encoded = self.encoder(x)
        return self.fc_mean(_encoded), self.fc_logvar(_encoded)

    @torch.no_grad()
    def generate(self, z):
        return self.decode(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var



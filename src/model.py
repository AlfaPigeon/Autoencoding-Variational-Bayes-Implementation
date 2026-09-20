import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, input_dim1=3, input_dim2=128, hidden_dim=32, latent_dim=128, kernel_size=4):
        super(VAE, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        # Shape Calculatios===
        self.conv_out_size = input_dim2 - kernel_size + 1
        self.flattened_dim = self.conv_out_size * self.conv_out_size * hidden_dim 
        self.hidden_dim = hidden_dim
        # ====================

        self.conv_encode = nn.Conv2d(input_dim1, hidden_dim, kernel_size=kernel_size)
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, self.flattened_dim)
        self.trans_conv = nn.ConvTranspose2d(hidden_dim, input_dim1, kernel_size=kernel_size)

    def reparameterize(self, _mean, _logvar):
        std = torch.exp(0.5*_logvar)
        e = torch.randn_like(std)
        return _mean + e*std

    def decode(self, z):

        hidden = torch.relu(self.fc_z(z))
        hidden = hidden.view(-1, self.hidden_dim, self.conv_out_size, self.conv_out_size)
        # Reshape into 4d        

        #recon = recon.view(-1, self.input_dim1, self.input_dim2, self.input_dim2)

        recon = self.trans_conv(hidden)
        recon = torch.sigmoid(recon)
        return recon
        
    def encode(self, x):
        h1 = torch.relu(
            self.conv_encode(x)
        )
        flat_h1 =  self.flatten(h1)
        return self.fc_mean(flat_h1), self.fc_logvar(flat_h1)

    @torch.no_grad()
    def generate(self, z):
        return self.decode(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var



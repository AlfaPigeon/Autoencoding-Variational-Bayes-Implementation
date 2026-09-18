import torch
import torch.nn as nn

class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, latent_dim=32, kernel_size=4):
        super(VAE, self).__init__()

        self.conv_encode = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size)
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(hidden_dim // kernel_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // kernel_size, latent_dim)
        self.fc_decode = nn.Linear(input_dim,input_dim)


    def encode(self, x):
        h1 = self.conv_encode(x)
        flat_h1 =  nn.Flatten(h1)
        return self.fc_mean(flat_h1), self.fc_logvar(flat_h1)

    def reparameterize(self, _mean, _logvar):
        std = torch.exp(0.5*_logvar)
        e = torch.randn_like(std)
        return _mean + e*std

    def decode(self, z):
        h3 = self.fc3(z)
        return torch.sigmoid(h3)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var


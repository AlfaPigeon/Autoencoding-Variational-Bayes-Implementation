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

        print("Flattened Dim: ", self.flattened_dim)

        self.conv_encode = nn.Conv2d(input_dim1, hidden_dim, kernel_size=kernel_size)
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, input_dim1*input_dim2*input_dim2)

    def reparameterize(self, _mean, _logvar):
        std = torch.exp(0.5*_logvar)
        e = torch.randn_like(std)
        return _mean + e*std

    def decode(self, z):
        recon = torch.sigmoid(
            self.fc_z(z)
        )
        # Reshape into 4d

        print(recon.shape)
        

        recon = recon.view(-1, self.input_dim1, self.input_dim2, self.input_dim2)
        return recon
        
    def encode(self, x):
        h1 = torch.relu(
            self.conv_encode(x)
        )
        print("Conv_Output_Shape: ", h1.shape)
        flat_h1 =  self.flatten(h1)
        print("Flat_Shape: ",flat_h1.shape)
        return self.fc_mean(flat_h1), self.fc_logvar(flat_h1)

    @torch.no_grad()
    def generate(self, z):
        return self.decode(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var


'''

Flattened Dim:  76832
VAE(
  (conv_encode): Conv2d(3, 32, kernel_size=(32, 32), stride=(1, 1))
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_mean): Linear(in_features=76832, out_features=128, bias=True)
  (fc_logvar): Linear(in_features=76832, out_features=128, bias=True)
  (fc_z): Linear(in_features=128, out_features=3, bias=True)
)
Conv_Output_Shape:  torch.Size([32, 32, 97, 97])
Flat_Shape:  torch.Size([32, 301088])

RuntimeError: Failed to run torchinfo. See above stack traces for more details. Executed layers up to: [Conv2d: 1, Flatten: 1]

'''
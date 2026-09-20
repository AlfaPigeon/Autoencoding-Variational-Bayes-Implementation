import torch
import matplotlib.pyplot as plt

def KL(_mean, _logvar):
    return -0.5 * torch.mean(1 + _logvar - _mean.pow(2) - _logvar.exp())

def PlotReconstructions(batch, recon_batch, n=8):
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        # Removed the un-normalization math
        axes[0, i].imshow(batch[i].cpu().permute(1, 2, 0))
        axes[0, i].axis('off')
        
        # Added detach().cpu() so Matplotlib can read GPU tensors
        axes[1, i].imshow(recon_batch[i].detach().cpu().permute(1, 2, 0))
        axes[1, i].axis('off')
    plt.show()
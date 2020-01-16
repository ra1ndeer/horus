import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder module
        self.input2hidden = nn.Linear(self.input_size, self.hidden_size)
        self.hidden2mu = nn.Linear(self.hidden_size, self.latent_size)
        self.hidden2logvar = nn.Linear(self.hidden_size, self.latent_size)

        # Decoder module
        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size)
        self.hidden2output = nn.Linear(self.hidden_size, self.input_size)

    # Encode inputs
    def encode(self, x):
        h = torch.tanh(self.input2hidden(x))
        return self.hidden2mu(h), self.hidden2logvar(h)

    # Decode latent space
    def decode(self, z):
        h = torch.tanh(self.latent2hidden(z))
        return torch.sigmoid(self.hidden2output(h))

    # Reparametrization trick
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    # Forward pass over the network
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_prime = self.decode(z)
        return x_prime, mu, logvar 

    # Train for a single epoch
    def train_epoch(self, train_loader, optimizer):
        self.train()

        kl_out = 0
        recon_out = 0

        for (x, _) in train_loader:
            optimizer.zero_grad()
            x = x.view(-1, self.input_size)
            x_prime, mu, logvar = self.forward(x)
            kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - logvar - 1.)
            recon = F.binary_cross_entropy(x_prime, x, reduction="sum")
            loss = kl + recon

            kl_out += kl.item()
            recon_out += recon.item()
            loss.backward()
            optimizer.step()

        return kl_out / (len(train_loader.dataset)), recon_out / len(train_loader.dataset)

    # Train the model for some epochs
    def fit(self, train_loader, optimizer, num_epochs):
        kl = list()
        recon = list()
        epochs = list(range(1, num_epochs+1))
        
        for e in epochs:
            print(f"Training epoch {e}")
            a, b = self.train_epoch(train_loader, optimizer)
            kl.append(a)
            recon.append(b)

        kl = np.array(kl)
        recon = np.array(recon)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, kl, marker=".", linestyle="--", label="KL")
        ax.plot(epochs, recon, marker=".", linestyle="--", label="Recon")
        ax.plot(epochs, kl+recon, marker=".", linestyle="--", label="ELBO")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.show()
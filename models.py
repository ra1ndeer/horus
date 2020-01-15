import torch
import torch.nn as nn 


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

    def encode(self, x):
        h = torch.tanh(self.input2hidden(x))
        return self.hidden2mu(h), self.hidden2logvar(h)

    def decode(self, z):
        h = torch.tanh(self.latent2hidden(z))
        return torch.sigmoid(self.hidden2output(h))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_prime = self.decode(z)
        return x_prime, mu, logvar 
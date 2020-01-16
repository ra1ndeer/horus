import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class PlanarTransform(nn.Module):
    def __init__(self, input_size, init_scale=0.01):
        super().__init__()

        self.input_size = input_size 
        self.u = nn.Parameter(torch.randn(1, self.input_size)*init_scale)
        self.w = nn.Parameter(torch.randn(1, self.input_size)*init_scale)
        self.b = nn.Parameter(torch.randn(())*init_scale)

    def forward(self, z):
        utw = self.u @ self.w.t()
        w_norm = self.w / torch.norm(self.w, p=2)
        u_hat = (torch.log(1 + torch.exp(utw)) - 1 - utw) * w_norm + self.u
        tanh_arg = z @ self.w.t() + self.b 
        zk = z + u_hat * torch.tanh(tanh_arg)
        logdetjac = -torch.log(torch.abs((1 - torch.tanh(tanh_arg)**2) * self.w @ u_hat.t() + 1) + 1e-10) 
        return zk, logdetjac

    def backward(self, z):
        raise NotImplementedError("Planar transforms have no algebraic inverse.")



class RadialTransform(nn.Module):
    def __init__(self, input_size, init_scale=0.01):
        super().__init__() 

        self.input_size = input_size 
        self.a = nn.Parameter((torch.rand(())+1e-6)*init_scale)
        self.b = nn.Parameter(torch.randn(())*init_scale)
        self.z_ref = nn.Parametr(torch.randn(1, self.input_size)*init_scale)

    def forward(self, z):
        dif = z - self.zref 
        norm = torch.norm(dif, p=2)
        b = torch.log(1. + torch.exp(self.b)) - self.a
        frac = b * dif / (self.a + norm)
        logdetjac = torch.log(torch.abs(torch.pow(frac, self.input_size - 1) * (1. + (self.a * b) / torch.pow(self.a + norm, 2))))
        zk = z + frac 
        return zk, logdetjac

    def backward(self, z):
        raise NotImplementedError("Radial transforms have no algebraic inverse.")



class NormalizingFlow(nn.Module):
    def __init__(self, input_size, flow_size, flow_type):
        super().__init__()

        self.input_size = input_size
        self.flow_size = flow_size 
        self.flow_type = flow_type 

        if self.flow_type == "p":
            self.flow = nn.ModuleList([PlanarTransform(self.input_size) for _ in range(self.flow_type)])
        elif self.flow_type == "r":
            self.flow = nn.ModuleList([RadialTransform(self.input_size) for _ in range(self.flow_type)])
        else:
            raise NotImplementedError("Not an available transformation.")

    def forward(self, z):
        zk = z
        logdetjac = 0.
        for transform in self.flow:
            zk, ldj_term = transform(zk)
            logdetjac += ldj_term 
        return zk, logdetjac



class VAE_NF(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, flow_size, flow_type):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.latent_size = latent_size 
        self.flow_size = flow_size 
        self.flow_type = flow_type

        # Encoder module
        self.input2hidden = nn.Linear(self.input_size, self.hidden_size)
        self.hidden2mu = nn.Linear(self.hidden_size, self.latent_size)
        self.hidden2logvar = nn.Linear(self.hidden_size, self.latent_size)

        # Decoder module
        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size)
        self.hidden2output = nn.Linear(self.hidden_size, self.input_size)

        self.normalizing_flow = NormalizingFlow(self.latent_size, self.flow_size, self.flow_type)

    def encode(self, x):
        h = torch.tanh(self.input2hidden(x))
        return self.hidden2mu(h), self.hidden2logvar(h)

    def decode(self, z):
        h = torch.tanh(self.latent2hidden(z))
        return torch.sigmoid(self.hidden2output)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z0 = self.reparametrize(mu, logvar)
        zk, logdetjac = self.normalizing_flow(z0)
        base = torch.distributions.normal.Normal(mu, torch.exp(0.5*logvar))
        prior = torch.distributions.normal.Normal(0, 1)
        logprior = prior.log_prob(zk).sum(-1)
        logbase_0 = base.log_prob(z0).sum(-1)
        kl_div = logbase_k = logbase_0 + logdetjac.view(-1) - logprior
        x_prime = self.decode(zk)
        return x_prime, kl_div.sum()

    def train_epoch(self, train_loader, optimizer):
        self.train()
        kl_out = 0.
        recon_out = 0.

        for (x, _) in train_loader:
            optimizer.zero_grad()
            x = x.view(-1, self.input_size)
            x_prime, kl = self.forward(x)
            recon = F.binary_cross_entropy(x_prime, x, reduction="sum")
            loss = recon + kl
        
            kl_out += kl.item()
            recon_out += recon.item()
            loss.backward()
            optimizer.step()

        return kl_out / (len(train_loader.dataset)), recon_out / len(train_loader.dataset)



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

        kl_out = 0.
        recon_out = 0.

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
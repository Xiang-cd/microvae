import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image

# 定义 VAE 基类
class VAE(nn.Module):
    def __init__(self, latent_dim=40):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # 编码器
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(512, 28*28)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder(z)
        return self.fc_out(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

# 定义 Bernoulli VAE
class BernoulliVAE(VAE):
    def __init__(self, latent_dim=40):
        super(BernoulliVAE, self).__init__(latent_dim)
    
    def forward(self, x):
        z, mu, logvar = super().forward(x)
        x_recon_logits = self.decode(z)
        x_recon = torch.sigmoid(x_recon_logits)
        return x_recon, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar):
        BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
        # KL 散度
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# 定义 Gaussian VAE
class GaussianVAE(VAE):
    def __init__(self, latent_dim=40, sigma=1.0):
        super(GaussianVAE, self).__init__(latent_dim)
        self.sigma = sigma
    
    def forward(self, x):
        z, mu, logvar = super().forward(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar):
        MSE = F.mse_loss(x_recon, x, reduction='sum') / (self.sigma ** 2)
        # KL 散度
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

# 数据准备
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练函数
def train_vae(model, train_loader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data)
            loss = model.loss_function(data.view(-1, 28*28), x_recon, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        average_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}, Loss: {average_loss:.4f}')

# 采样函数
def sample_vae(model, device, save_path, num_samples=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        if isinstance(model, BernoulliVAE):
            x_recon_logits = model.decode(z)
            x_recon_probs = torch.sigmoid(x_recon_logits)
            x_recon = torch.bernoulli(x_recon_probs)  # 从伯努利分布中采样
        else:  # GaussianVAE
            x_recon = model.decode(z)
            x_recon = torch.normal(x_recon, model.sigma) # 从正态分布中采样
        x_recon = x_recon.view(-1, 1, 28, 28).cpu()
        grid = torchvision.utils.make_grid(x_recon, nrow=4)
        save_image(grid, save_path, nrow=4)

if __name__ == '__main__':
    # 训练和采样
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    bernoulli_vae = BernoulliVAE(latent_dim=40).to(device)
    gaussian_vae = GaussianVAE(latent_dim=40, sigma=1.0).to(device)

    # 优化器
    optimizer_b = optim.Adam(bernoulli_vae.parameters(), lr=1e-3)
    optimizer_g = optim.Adam(gaussian_vae.parameters(), lr=1e-3)

    # 训练 Bernoulli VAE
    print("Training Bernoulli VAE")
    train_vae(bernoulli_vae, train_loader, optimizer_b, device, epochs=10)

    # 从 Bernoulli VAE 采样
    print("\nSampling from Bernoulli VAE")
    sample_vae(bernoulli_vae, device, 'bernoulli.png', num_samples=16)
    # 训练 Gaussian VAE
    print("\nTraining Gaussian VAE")
    train_vae(gaussian_vae, train_loader, optimizer_g, device, epochs=10)

    # 从 Gaussian VAE 采样
    print("\nSampling from Gaussian VAE")
    gaussian_vae.sigma = 0.1
    sample_vae(gaussian_vae, device, 'gaussian_sigma0.1.png', num_samples=16)
    
    gaussian_vae.sigma = 0.0
    sample_vae(gaussian_vae, device, 'gaussian_sigma0.0.png', num_samples=16)
    
    gaussian_vae.sigma = 0.3
    sample_vae(gaussian_vae, device, 'gaussian_sigma0.3.png', num_samples=16)

import torch
import torch.nn as nn

class unFlatten(nn.Module):
    def forward(self,x):
        return x.view(x.size[0],3,8,8)

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.size[0],-1)

class VAE(nn.Module):
    '''
    - Dimension of latent vector Z is 5 
    - Assume that variational posterior is gaussian with independent of each dimension (covariance matrix is diagonal)
    '''
    def __init__(self):
        self.latentDimension = 5
        self.encoder = nn.Sequential(
            nn.Conv2d(3,6,3,2,1), # (66,16,16)
            nn.ReLU(),
            nn.Conv2d(6,12,3,2,1), # (3,8,8)
            nn.ReLU(),
            Flatten(), # 3*8*8 = 192
            nn.Linear(192,self.latentDimension*2) # mu, log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latentDimension,512),
            nn.ReLU(),
            unFlatten(),
            nn.ConvTranspose2d(12,6,3,2,1,1), # (6,16,16) 
            nn.ReLU(),
            nn.ConvTranspose2d(6,3,3,2,1,1), # (3,32,32)
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = image with (C,H,W). 
        out = self.encoder(x) # (5)
        mu,log_var = out[:,:self.latentDimension], out[:,self.latentDimension:]
        e = torch.randn_like(mu) # 
        z = mu + e * torch.sqrt(torch.exp(log_var))
        out = self.decoder(z)
        return mu,log_var,out
        

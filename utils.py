import torch

def loss(mu,log_var,out,x):
    kl_loss = torch.tensor(-1/2) * torch.sum(torch.exp(log_var) + torch.square(mu) - log_var -torch.ones_like(mu))
    reconstrut_loss = torch.sum((x- out)**2)
    return kl_loss + reconstrut_loss
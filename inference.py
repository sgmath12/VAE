import torch
from vae import *
import pdb
import utils


latent_dim = 2
batch_size = 1
num_images = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vae_MNIST()

model.load_state_dict(torch.load('./saved_models/saved_model.pth'))
model = model.to(device)
model.eval()

with torch.no_grad():
    for i in range(num_images):
        eps = torch.randn([batch_size,latent_dim])
        eps = eps.to(device)
        out = model.decoder(eps).cpu().numpy() #[batch_size,1,28,28]
        utils.saveImg(out[0],i)

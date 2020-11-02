import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import utils
from vae import *
import pdb
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.ToTensor()]
    )

path = os.path.expanduser( '~/data/MNIST')
model_save_path = os.path.expanduser('./saved_models')
trainset = torchvision.datasets.MNIST(root = path, train=False,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size= 16,shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vae_MNIST()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

for epoch in range(30):
    running_loss = 0.0
    for i,(x,y) in tqdm(enumerate(trainloader)):
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        mu,log_var,out = model(x)
        loss = utils.loss(mu,log_var,out,x)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(loss.item())

# save the model
torch.save(model.state_dict(),model_save_path+'model.pth')
print ("-"*15)
print ("End Training")
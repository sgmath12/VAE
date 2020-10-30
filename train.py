import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import utils
from vae import *
import pdb

transform = transforms.Compose(
    [transforms.ToTensor()]
    )

path = os.path.expanduser( '~/data/MNIST')

trainset = torchvision.datasets.CIFAR10(root = path, train=True,
                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True)
model = vae_CIFAR()
optimizer = optim.SGD(model.parameters(), lr = 0.0001)

for epoch in range(100):
    running_loss = 0.0
    for i,(x,y) in enumerate(trainloader):
        mu,log_var,out = model(x)
        optimizer.zero_grad()
        loss = utils.loss(mu,log_var,out,x)
        loss.backward()
        optimizer.step()
        # print (loss)
        running_loss += loss.item()
        if i % 500 == 0 :
            print(running_loss/500)
            running_loss = 0
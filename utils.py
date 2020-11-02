import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt

def loss(mu,log_var,out,x):
    N = mu.shape[0]
    kl_loss = torch.mean(0.5 *torch.sum( torch.exp(log_var) + torch.square(mu) - log_var -1,dim = 1),dim = 0)
    reconstruct_loss = torch.mean(torch.sum((x- out)**2))
    return kl_loss + reconstruct_loss

def array2image(arr):
    
    arr = np.clip(arr,0,1) * 255
    pdb.set_trace()
    img = Image.fromarray(np.uint8(arr))
    return img

def saveImg(arr,i):
    arr = arr.reshape([28,28])
    arr = np.clip(arr,0,1)
    plt.imsave('./images/sample' + str(i) + '.png',arr,cmap = 'Greys')

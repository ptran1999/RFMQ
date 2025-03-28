#!/usr/bin/env python
# coding: utf-8

# In[1]:


%load_ext autoreload
%autoreload 2

# In[35]:


import importlib
import rfm_q
importlib.reload(rfm_q)  # Reload the updated module
from rfm_q import q_rfm



# In[36]:


from qiskit import QuantumCircuit
from qiskit_aer import Aer
print(Aer.backends())


# In[37]:


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# In[38]:


# set data path
def set_data_path():
    return "../data/"
#     raise NotImplementedError

# In[39]:


def pre_process(torchset,n_samples,num_classes=10):
    indices = list(np.random.choice(len(torchset),n_samples))

    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
    return trainset

# In[40]:


# load svhn data
transform = transforms.Compose([
    transforms.ToTensor()
])

data_path = set_data_path() ## set this data path

trainset0 = torchvision.datasets.SVHN(root=data_path,
                                    split = "train",
                                    transform=transform,
                                    download=True)
testset0 = torchvision.datasets.SVHN(root=data_path,
                                    split = "test",
                                    transform=transform,
                                    download=True)

trainset = pre_process(trainset0,n_samples=5000, num_classes=10)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)


testset = pre_process(testset0,n_samples=5000, num_classes=10)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)



# In[44]:


# rfm import 
from rfm import *

# In[53]:


pip install qiskit-aer

# In[54]:


import importlib
import rfm_q
importlib.reload(rfm_q)
from rfm_q import q_rfm


# In[55]:


import qiskit
print(qiskit.__qiskit_version__)


# In[50]:



from rfm_q import q_rfm
M, _ = q_rfm(train_loader, test_loader, iters=2, loader=True, classif=True)



# In[14]:


from qiskit_machine_learning.kernels import BaseKernel
from qiskit_aer import AerSimulator


# In[18]:


# run rfm
M, _ = q_rfm(train_loader, test_loader, iters=2, loader=True, classif=True)

# We have run three steps of RFM (the first iterate is the original laplace kernel), returning the M matrix of the final iterate.

# In[14]:


# visualize M matrix
import matplotlib.pyplot as plt
%matplotlib inline

# In[15]:


f, axarr = plt.subplots(1,2,figsize=(10, 3))
axarr[0].axes.xaxis.set_ticklabels([])
axarr[0].axes.yaxis.set_ticklabels([])
axarr[1].axes.xaxis.set_ticklabels([])
axarr[1].axes.yaxis.set_ticklabels([])

pcm = axarr[0].imshow(np.mean(np.diag(M).reshape(3,32,32),axis=0),cmap='cividis')
axarr[0].set_title("M matrix diagonal")
f.colorbar(mappable=pcm, ax=axarr[0], shrink=0.8,location="left")
axarr[1].imshow(torch.moveaxis(trainset0[3][0],0,2))
axarr[1].set_title("Sample Image")
print()

# Here we plot the diagonal (averaged across channels) of the M matrix to see which coordinates are being highlighted by RFM. The center of the image is highlighted (where the digits appear). 

# In[ ]:




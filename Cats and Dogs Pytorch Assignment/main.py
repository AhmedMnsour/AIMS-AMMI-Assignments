from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset import CatDogDataset
from model import CNN
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim


def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

input_size = 224 * 244 * 3  
output_size = 2  

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize((224, 224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])



path    = 'dataset/train'
dataset = CatDogDataset(path, transform=transform)


shuffle     = True
batch_size  = 64
num_workers = 0
dataloader  = DataLoader(dataset=dataset, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)



accuracy_list = []
n_features = 6  

model_cnn = CNN(input_size, n_features, output_size)
optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_cnn)))


def train(epoch, model):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            #print('k')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))







if __name__ == "__main__":
    for epoch in range(0, 1):
        train(epoch, model_cnn)




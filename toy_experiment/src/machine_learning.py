#Torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

from tqdm import tqdm
import pandas as pd

def fwd_pass_classifier(net, X:Tensor, y:Tensor, device, optimizer, train=False):
    """
    This function controls the machine learning steps, depending on if we are in training mode or not.
    """
    if train:
        net.train()
        net.zero_grad()
    outputs = net(X.to(device))
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def train_classifier(net, traindata, testdata, batchsize:int, epochs:int, device, optimizer):
    """
    Trains the model for the number of epochs specified, using the batch size specified.
    Returns a dataframe with the stats from the training.
    """
    dataset = DataLoader(traindata, batchsize, shuffle=True)
    df_labels = ["Loss", "Accuracy", "Validation loss", "Validation accuracy", "Epoch", "Iteration"]
    df_data = [[0], [0], [0], [0], [0], [0]]
    df = pd.DataFrame(dict(zip(df_labels, df_data)))
    i = 0
    for epoch in tqdm(range(epochs)):
        for data in dataset:
            i = i+1
            X, y = data
            #print(X[0], y[0])
            acc, loss = fwd_pass_classifier(net, X, y, device, optimizer, train=True)
            #acc, loss = test(net, testdata, size=size)
            if i % 10 == 0:
                val_acc, val_loss = test_classifier(net, testdata, device, optimizer, batchsize)
                df_data = [float(loss), acc, float(val_loss), val_acc, epoch, i]
                new_df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                df = pd.concat([df, new_df], ignore_index=True)
            
    return df

def test_classifier(net, data, device, optimizer, size:int = 32):
    """
    Calculates the accuracy and the loss of the model for a random batch.
    """
    net.eval()
    dataset = DataLoader(data, size, shuffle=True) #shuffle data and choose batch size
    X, y = next(iter(dataset)) #get a random batch
    val_acc, val_loss = fwd_pass_classifier(net, X, y, device, optimizer, train=False)
    return val_acc, val_loss
    
def predict_classifier(net, testdata, num_classes:int, size:int, device):
    """
    Calculates the accuracy and the loss of the model in testing mode.
    If return_loss is True, it will return the loss for each datapoint.
    It can also return the softmax values of the raw output from the model.
    Does not shuffle the data.
    """
    dataset = DataLoader(testdata, size, shuffle=False) #shuffle data and choose batch size
    logits = torch.zeros((len(dataset), size, num_classes))
    truth = torch.zeros((len(dataset), size))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X, y = data
            logits[i] = net(X.to(device))
            truth[i] = torch.argmax(y, dim=-1).to(torch.int)
            i = i+1
    return torch.flatten(truth), logits
#Torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

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

def train_classifier(net, traindata, testdata, batchsize:int, epochs:int, device, optimizer, early_stopping:int=-1):
    """
    Trains the model for the number of epochs specified, using the batch size specified.
    Returns a dataframe with the stats from the training.
    """
    dataset = DataLoader(traindata, batchsize, shuffle=True)
    df_labels = ["Loss", "Accuracy", "Validation loss", "Validation accuracy", "Epoch", "Iteration"]
    df_data = [[1], [0], [1], [0], [0], [0]]
    df = pd.DataFrame(dict(zip(df_labels, df_data)))
    i = 0
    patience = early_stopping #How many epochs to keep training if no improvement in validation loss
    min_loss = None
    for epoch in tqdm(range(epochs)):
        for data in dataset:
            i = i+1
            X, y = data
            #print(X[0], y[0])
            acc, loss = fwd_pass_classifier(net, X, y, device, optimizer, train=True)
            #acc, loss = test(net, testdata, size=size)
            if i % 10 == 0: #Record every ten batches
                val_acc, val_loss = test_classifier(net, testdata, device, optimizer, batchsize)
                df_data = [float(loss), acc, float(val_loss), val_acc, epoch, i]
                new_df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                df = pd.concat([df, new_df], ignore_index=True)
        if ((early_stopping > 0) and (len(df) > 1)): #If small data, we might not have validation loss yet
            if min_loss == None:
                min_loss = float(val_loss)
            elif min_loss <= df["Validation loss"].min():
                patience = patience - 1
            elif min_loss > df["Validation loss"].min():
                min_loss = df["Validation loss"].min()
                patience = early_stopping # Restart early_stopping
            if patience == 0:
                print(f"Stopping training early at epoch {epoch}")
                df.drop([0])
                return df
    df.drop([0])
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
    assert len(testdata)%size==0, "Please choose batch size so that testdata%size==0."

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
    return torch.flatten(truth), logits.view(-1, num_classes)

def enable_dropout(net):
    """ Function to enable the dropout layers during test-time """
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_predictions(net,
                   testdata,
                   batchsize,
                   forward_passes,
                   n_classes,
                   n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes
    #https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    net : object
        ML model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """
    data_loader = DataLoader(testdata, batchsize, shuffle=False)
    dropout_predictions = np.empty((0, n_samples, n_classes))
    print("Starting MC dropout inference: ")
    for i in tqdm(range(forward_passes)):
        predictions = np.empty((0, n_classes))
        net.eval()
        enable_dropout(net)
        for i, (X, y) in enumerate(data_loader):
            X = X.to(torch.device('cuda'))
            with torch.no_grad():
                output = net(X)
                output = torch.softmax(output, dim=-1)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    #epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    #entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    #mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),
    #                                       axis=-1), axis=0)  # shape (n_samples,)
    return mean, variance

# Make one hot vectors
def label_maker(values, num_classes):
    labels = np.zeros((len(values), num_classes))
    for i, value in enumerate(values):
        labels[i][value] = 1
    return torch.Tensor(labels).to(torch.int)
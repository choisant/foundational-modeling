#Torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

import torchbnn as bnn

from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

def fwd_pass_classifier(net, X:Tensor, y:Tensor, device, optimizer, train:bool=False, biased_class:int=-1, bias_weight:float=0.1):
    """
    This function controls the machine learning steps, depending on if we are in training mode or not.
    biased_class = -1 gives clean cross entropy without conflictual loss term
    """
    if train:
        net.train()
        net.zero_grad()
    outputs = net(X.to(device))
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    #Should definitely make this stricter
    if biased_class == -1:
        loss = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device)) 
    else:
        weight = bias_weight
        ce_loss = F.cross_entropy(outputs, torch.argmax(y,dim=-1).to(device))
        biased_class_labels = label_maker([int(biased_class)]*len(outputs), 2).to(device)
        #torch.Tensor([biased_class]*len(outputs)).to(torch.int).to(device)
        bias_term = weight*F.cross_entropy(outputs, biased_class_labels.to(torch.float32)).to(device)
        loss = ce_loss + bias_term
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def train_classifier(net, traindata, testdata, batchsize:int, epochs:int, device, optimizer, 
                     early_stopping:int=-1, biased_class:int=-1, bias_weight:float=0.1, mcd:bool=False):
    """
    Trains the model for the number of epochs specified, using the batch size specified.
    Returns a dataframe with the stats from the training.
    """
    dataset = DataLoader(traindata, batchsize, shuffle=True)
    df_labels = ["Loss", "Accuracy", "Validation loss", "Validation accuracy", "Epoch", "Iteration"]
    df_created = False
    i = 0
    patience = early_stopping #How many epochs to keep training if no improvement in validation loss
    min_loss = None
    for epoch in tqdm(range(epochs)):
        # Iterate over batches
        for data in dataset:
            i = i+1
            X, y = data
            #print(X[0], y[0])
            acc, loss = fwd_pass_classifier(net, X, y, device, optimizer, train=True, 
                                            biased_class=biased_class, bias_weight=bias_weight)
            #acc, loss = test(net, testdata, size=size)
            if i % 10 == 0: #Record every ten batches
                val_acc, val_loss = test_classifier(net, testdata, device, optimizer, batchsize, 
                                                    biased_class=biased_class, bias_weight=bias_weight, mcd=mcd)
                df_data = [float(loss), acc, float(val_loss), val_acc, epoch, i]
                if df_created == False:
                    df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                    df_created = True
                else:
                    new_df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                    df = pd.concat([df, new_df], ignore_index=True)
        #Check every epoch if we should stop
        if ((early_stopping > 0) and df_created): #If small data, we might not have validation loss yet
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

def test_classifier(net, data, device, optimizer, size:int = 32, biased_class:int=-1, 
                    bias_weight:float=0.1, mcd:bool=False):
    """
    Calculates the accuracy and the loss of the model for a random batch.
    """
    net.eval()
    if mcd:
        enable_dropout(net)
    dataset = DataLoader(data, size, shuffle=True) #shuffle data and choose batch size
    X, y = next(iter(dataset)) #get a random batch
    val_acc, val_loss = fwd_pass_classifier(net, X, y, device, optimizer, train=False, 
                                            biased_class=biased_class, bias_weight=bias_weight)
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
        for data in dataset:
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
                   device,
                   testdata,
                   forward_passes,
                   n_classes,
                   n_samples,
                   return_samples:bool = False):
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
    return_MCsamples : bool
        return array of MC samples. Default false
    """
    data_loader = DataLoader(testdata, n_samples, shuffle=False)
    dropout_predictions = np.empty((0, n_samples, n_classes))
    print("Starting MC dropout inference: ")
    for i in tqdm(range(forward_passes)):
        predictions = np.empty((0, n_classes))
        net.eval()
        enable_dropout(net)
        for i, (X, y) in enumerate(data_loader): #Just one "batch"
            X = X.to(device)
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
    # 
    #                                       axis=-1), axis=0)  # shape (n_samples,)
    if return_samples:
        return mean, variance, dropout_predictions
    else:
        return mean, variance

# Make one hot vectors
def label_maker(values, num_classes):
    labels = np.zeros((len(values), num_classes))
    for i, value in enumerate(values):
        labels[i][value] = 1
    return torch.Tensor(labels).to(torch.int)


#########################
# Bayesian Neural network
#########################

def fwd_pass_bnn_classifier(net, X:Tensor, y:Tensor, device, optimizer, epoch:int, max_epoch:int, batchsize:int, train=False):
    """
    This function controls the machine learning steps, depending on if we are in training mode or not.
    """
    if train:
        net.train()
        net.zero_grad()
    outputs = net(X.to(device))
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 10/batchsize #Not sure if this is too low...
    #kl_weight = 10/batchsize
    #annealing_coef = min(1.0, epoch / max_epoch)
    ce = ce_loss(outputs, torch.argmax(y,dim=-1).to(device))
    kl = kl_loss(net)
    cost = (ce + kl_weight*kl).to(device)
    if train:
        cost.backward()
        optimizer.step()
    return acc, ce, kl, cost

def train_bnn_classifier(net, traindata, testdata, batchsize:int, epochs:int, device, optimizer, early_stopping:int=-1):
    """
    Trains the model for the number of epochs specified, using the batch size specified.
    Returns a dataframe with the stats from the training.
    """
    dataset = DataLoader(traindata, batchsize, shuffle=True)
    df_labels = ["CE loss", "KL loss", "Cost", "Accuracy", 
                 "Validation CE loss", "Validation KL loss",
                 "Validation cost", "Validation accuracy", 
                 "Epoch", "Iteration"]
    df_created = False # Bool to check if df is created
    #df = pd.DataFrame(dict(zip(df_labels, df_data)))
    i = 0
    patience = early_stopping #How many epochs to keep training if no improvement in validation loss
    min_loss = None
    for epoch in tqdm(range(epochs)):
        # Iterate over batches
        for data in dataset:
            i = i+1
            X, y = data
            #print(X[0], y[0])
            acc, ce_loss, kl_loss, cost = fwd_pass_bnn_classifier(net, X, y, device, optimizer, epoch, max_epoch=epochs, batchsize=batchsize, train=True)
            #acc, loss = test(net, testdata, size=size)
            if i % 10 == 0: #Record every ten batches
                val_acc, val_ce_loss, val_kl_loss, val_cost = test_bnn_classifier(net, testdata, device, optimizer, epoch, max_epoch=epochs, batchsize=batchsize)
                df_data = [float(ce_loss), float(kl_loss), float(cost), acc, 
                           float(val_ce_loss), float(val_kl_loss), float(val_cost), val_acc, 
                           epoch, i]
                if df_created == False:
                    df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                    df_created = True
                else:
                    new_df = pd.DataFrame(dict(zip(df_labels, df_data)), index=[0])
                    df = pd.concat([df, new_df], ignore_index=True)
        #Check every epoch if we should stop
        if ((early_stopping > 0) and df_created): #If small data, we might not have validation loss yet
            if min_loss == None:
                min_loss = float(val_cost)
            elif min_loss <= df["Validation cost"].min():
                patience = patience - 1
            elif min_loss > df["Validation cost"].min():
                min_loss = df["Validation cost"].min()
                patience = early_stopping # Restart early_stopping
            if patience == 0:
                print(f"Stopping training early at epoch {epoch}")
                df.drop([0])
                return df
    df.drop([0])
    return df

def test_bnn_classifier(net, data, device, optimizer, epoch:int, max_epoch:int, batchsize:int = 32):
    """
    Calculates the accuracy and the loss of the model for a random batch.
    """
    net.eval()
    dataset = DataLoader(data, batchsize, shuffle=True) #shuffle data and choose batch size
    X, y = next(iter(dataset)) #get a random batch
    val_acc, val_ce_loss, val_kl_loss, val_cost = fwd_pass_bnn_classifier(net, X, y, device, optimizer, epoch, max_epoch, batchsize, train=False)
    return val_acc, val_ce_loss, val_kl_loss, val_cost
    
def predict_bnn_classifier(net, testdata, num_classes:int, size:int, device):
    """
    Returns a list of predictions and truth values on testdata.
    It can also return the softmax values of the raw output from the model.
    Does not shuffle the data.
    """
    assert len(testdata)%size==0, "Please choose batch size so that testdata%size==0."

    dataset = DataLoader(testdata, size, shuffle=False) #shuffle data and choose batch size
    probs = torch.zeros((len(dataset), size, num_classes))
    preds = torch.zeros((len(dataset), size))
    truth = torch.zeros((len(dataset), size))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X, y = data
            output = net(X.to(device))
            probs[i] = torch.softmax(output.data, dim=-1)
            
            _, predicted = torch.max(output.data, 1)
            preds[i] = predicted
            truth[i] = torch.argmax(y, dim=-1).to(torch.int)
            i = i+1
    return torch.flatten(preds), torch.flatten(truth), probs.view(-1, num_classes)


def predict_bnn(model, dataset, df, device, n_classes=2, n_samples=100):
    df_new = df.copy()
    dataset = DataLoader(dataset, len(dataset), shuffle=False)
    preds = torch.zeros((n_samples, len(df)))
    probs = torch.zeros((n_samples, len(df), n_classes))

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_samples)):
            for data in dataset:
                X, y = data
                y = torch.Tensor.float(y)
                pre = model(X.to(device))
                _, predicted = torch.max(pre.data, 1)
                preds[i] = predicted
                probs[i] = torch.softmax(pre.data, dim=-1)

    df_new["Prediction_median"] = preds.median(axis=0)[0]
    df_new["Prediction_mean"] = preds.mean(axis=0)
    
    df_new["Est_prob_blue"] = probs[:,:,1].mean(axis=0)
    df_new["Std_prob_blue"] = probs[:,:,1].std(axis=0)

    df_new["Prediction"] = torch.argmax(probs.mean(axis=0), dim=-1)
    return df_new
#Torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor

import torchbnn as bnn

from tqdm import tqdm
import pandas as pd
#https://github.com/clabrugere/evidential-deeplearning/tree/main

from edl_pytorch import Dirichlet, evidential_classification


class TypeIIMaximumLikelihoodLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Dirichlet distribution D(p|alphas) is used as prior on the likelihood of Multi(y|p)."""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=-1, keepdim=True)

        loss = torch.sum(labels * (torch.log(strength) - torch.log(alphas)), dim=-1)

        return torch.mean(loss)


class CEBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Bayes risk is the maximum cost of making incorrect estimates, taking a cost function assigning a penalty of
        making an incorrect estimate and summing it over all possible outcomes. Here the cost function is the Cross Entropy.
        """
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strengths = torch.sum(alphas, dim=-1, keepdim=True)

        loss = torch.sum(labels * (torch.digamma(strengths) - torch.digamma(alphas)), dim=-1)

        return torch.mean(loss)


class SSBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Same as CEBayesRiskLoss but here the cost function is the sum of squares instead."""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=-1, keepdim=True)
        probabilities = alphas / strength

        error = (labels - probabilities) ** 2
        variance = probabilities * (1.0 - probabilities) / (strength + 1.0)

        loss = torch.sum(error + variance, dim=-1)

        return torch.mean(loss)


class KLDivergenceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Acts as a regularization term to shrink towards zero the evidence of samples that cannot be correctly classified"""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        num_classes = evidences.size(-1)
        alphas = evidences + 1.0
        alphas_tilde = labels + (1.0 - labels) * alphas
        strength_tilde = torch.sum(alphas_tilde, dim=-1, keepdim=True)

        # lgamma is the log of the gamma function
        first_term = (
            torch.lgamma(strength_tilde)
            - torch.lgamma(evidences.new_tensor(num_classes, dtype=torch.float32))
            - torch.sum(torch.lgamma(alphas_tilde), dim=-1, keepdim=True)
        )
        second_term = torch.sum(
            (alphas_tilde - 1.0) * (torch.digamma(alphas_tilde) - torch.digamma(strength_tilde)), dim=-1, keepdim=True
        )
        loss = torch.mean(first_term + second_term)

        return loss
    

def fwd_pass_evidential_classifier(net, X:Tensor, y:Tensor, device, optimizer, batchsize:int, epoch:int, max_epoch, train:bool=False, Teddy:bool=False):
    """
    This function controls the machine learning steps, depending on if we are in training mode or not.
    """
    if train:
        net.train()
        net.zero_grad()
    outputs = net(X.to(device))
    #evidences = F.softplus(outputs)
    evidences = outputs
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    annealing_coef = min(1, epoch / max_epoch)/10
    #coef=0.001
    bayes_risk = SSBayesRiskLoss()
    br = bayes_risk(evidences, y.to(device)).to(device)
    kld_loss = KLDivergenceLoss()
    kld = annealing_coef*kld_loss(evidences, y.to(device)).to(device)

    if Teddy:
        loss = evidential_classification(outputs, y.to(device), lamb=annealing_coef)
    else:
        loss = br + kld
    
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss, br, kld

def train_evidential_classifier(net, traindata, testdata, batchsize:int, epochs:int, device, optimizer, early_stopping:int=-1, Teddy:bool=False):
    """
    Trains the model for the number of epochs specified, using the batch size specified.
    Returns a dataframe with the stats from the training.
    """
    dataset = DataLoader(traindata, batchsize, shuffle=True)
    df_labels = ["Loss", "Accuracy", "Bayes risk", "KLD loss", "Validation loss", "Validation accuracy", 
                 "Validation Bayes risk", "Validation KLD loss", "Epoch", "Iteration"]
    df_created = False
    i = 0
    patience = early_stopping #How many epochs to keep training if no improvement in validation loss
    min_loss = None
    for epoch in tqdm(range(epochs)):
        # Iterate over batches
        for data in dataset:
            i = i+1
            X, y = data
            acc, loss, br, kld = fwd_pass_evidential_classifier(net, X, y, device, optimizer, batchsize, epoch, max_epoch=epochs, train=True)
            if i % 10 == 0: #Record every ten batches
                val_acc, val_loss, val_br, val_kld = test_evidential_classifier(net, testdata, device, optimizer, batchsize, epoch, max_epoch=epochs, size=batchsize)
                df_data = [float(loss), acc, float(br), float(kld), float(val_loss), val_acc, float(val_br), float(val_kld), epoch, i]
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

def test_evidential_classifier(net, data, device, optimizer, batchsize:int, epoch, max_epoch, size:int = 32, Teddy:bool=False):
    """
    Calculates the accuracy and the loss of the model for a random batch.
    """
    net.eval()
    dataset = DataLoader(data, size, shuffle=True) #shuffle data and choose batch size
    X, y = next(iter(dataset)) #get a random batch
    val_acc, val_loss, val_br, val_kld = fwd_pass_evidential_classifier(net, X, y, device, optimizer, batchsize, epoch, max_epoch, train=False)
    return val_acc, val_loss, val_br, val_kld
    
def predict_evidential_classifier(net, testdata, num_classes:int, size:int, device, Teddy:bool=False):
    """
    Calculates the accuracy and the loss of the model in testing mode.
    If return_loss is True, it will return the loss for each datapoint.
    It can also return the softmax values of the raw output from the model.
    Does not shuffle the data.
    Teddy produces the alphas directly.
    """
    assert len(testdata)%size==0, "Please choose batch size so that testdata%size==0."

    dataset = DataLoader(testdata, size, shuffle=False) #shuffle data and choose batch size
    logits = torch.zeros((len(dataset), size, num_classes))
    evidences = torch.zeros((len(dataset), size, num_classes))
    alphas = torch.zeros((len(dataset), size, num_classes))
    strength = torch.zeros((len(dataset), size, num_classes))
    probabilities = torch.zeros((len(dataset), size, num_classes))
    uncertainties = torch.zeros((len(dataset), size, num_classes))
    i = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            X, y = data
            if Teddy:
                alphas[i] = net(X.to(device))
            else:
                logits[i] = net(X.to(device))
                evidences[i] = logits[i] #F.softplus(logits[i])
                alphas[i] = evidences[i] + 1.0
            strength[i] = torch.sum(alphas[i], dim=-1, keepdim=True) #Summing can go wrong
            probabilities[i] = alphas[i] / strength[i]
            if (probabilities[i].max() > 1 or probabilities[i].max() < 0):
                print("Something is wrong!")
                print("alphas: ", alphas)
                print("Strength: ", strength)
            uncertainties[i] = torch.sqrt(probabilities[i]*(1-probabilities[i])/(strength[i]+1))
            i = i+1
    total_uncertainty = num_classes / strength
    beliefs = evidences / strength
    if Teddy:
        return probabilities.view(-1, num_classes), uncertainties.view(-1, num_classes)
    else:
        return probabilities.view(-1, num_classes), uncertainties.view(-1, num_classes), beliefs.view(-1, num_classes)

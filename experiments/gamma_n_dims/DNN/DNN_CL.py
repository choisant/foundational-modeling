import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.optim as optim
from torchmetrics.classification import BinaryCalibrationError

from sklearn.metrics import roc_auc_score as roc_auc_score
from scipy.special import kl_div as kl_div
from sklearn.metrics import log_loss as log_loss

# import custom functions from src folder
module_path = str(Path.cwd() / "../../../src")

if module_path not in sys.path:
    sys.path.append(module_path)

from SequentialNet import SequentialNet
from machine_learning import train_classifier, predict_classifier
from util import label_maker

# Set up device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device {torch.cuda.get_device_name(1)}")

# Machine learning options
x1_key = "x1"
x2_key = "x2"
n_data = [250, 500, 1000, 2000, 3000, 5000, 10000]
bs_list = [128, 128, 128*2, 128*2, 1024, 1024, 1024*2]
#n_data = [10000]
max_err_val = [0]*len(n_data)
for i in range(len(n_data)):
    if n_data[i] < 1000:
        max_err_val[i] = 0.15
    else:
        max_err_val[i] = 0.1
patience = 30

#Data constants
shapes = [2, 4]
scales = [3, 3]
k = len(scales) # Number of classes
d = 2 # Number of dimensions
p_c = [1/len(shapes)]*len(shapes) # Uniform distributon over classes

tag = f'k_{k}_d{d}_shapes{shapes}_scales{scales}_pc{p_c}'.replace(" ", "")

# Read files
train_n = 50000
trainfile = f"train_n_{train_n}_{tag}"
valfile = f"val_n_5000_{tag}"
testfile = f"test_n_10000_{tag}"
gridfile = f"grid_x1_x2_10000_{tag}"

train_data = pd.read_csv(f"../data/{trainfile}.csv")
val_data = pd.read_csv(f"../data/{valfile}.csv")
test_data = pd.read_csv(f"../data/{testfile}.csv")
grid_data = pd.read_csv(f"../data/{gridfile}.csv")
grid_rmax = grid_data["x1"].max()

X_train = torch.Tensor(np.dstack((train_data[x1_key], train_data[x2_key]))).to(torch.float32)[0]
Y_train = label_maker(train_data["class"], 2)

X_val = torch.Tensor(np.dstack((val_data[x1_key], val_data[x2_key]))).to(torch.float32)[0]
Y_val = label_maker(val_data["class"], 2)

X_test = torch.Tensor(np.dstack((test_data[x1_key], test_data[x2_key]))).to(torch.float32)[0]
Y_test = label_maker(test_data["class"], 2)

X_grid = torch.Tensor(np.dstack((grid_data[x1_key], grid_data[x2_key]))).to(torch.float32)[0]
Y_grid = torch.zeros(X_grid.shape)

# Create datasets for pytorch
val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
grid_dataset = torch.utils.data.TensorDataset(X_grid, Y_grid)



def train_ensemble(n_ensemble, n_train, batchsize, n_classes):
    if (n_ensemble%n_classes) != 0:
        print("Please set n_ensembles to n_classes*int.")
        return None

    val_df = pd.read_csv(f"../data/{valfile}.csv")
    test_df = pd.read_csv(f"../data/{testfile}.csv")
    grid_df = pd.read_csv(f"../data/{gridfile}.csv")
    biased_class = 0
    # Timer
    start = timer()
    print(f"Starting training of {n_ensemble} ensembles with {n_train} training points.")
    for i in tqdm(range(n_ensemble)):
        print(f"Ensemble nr {i}, N train = {n_train}. Biased class: {biased_class}")

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train[0:n_train], Y_train[0:n_train])

        # Create new model
        model = SequentialNet(L=200, n_hidden=3, activation="relu", in_channels=2, out_channels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        training_results = train_classifier(model, train_dataset, 
                                val_dataset, batchsize=batchsize, epochs = 200, 
                                device = device, optimizer = optimizer, early_stopping=patience,
                                biased_class=biased_class)
        
         # Predict on validation set
        truth_val, logits_val = predict_classifier(model, val_dataset, 2, 100, device)
        preds_val = torch.argmax(logits_val, dim=-1).flatten()
        val_df[f"Prediction_{i}"] = preds_val
        val_df[f"Confidence_{i}"] = torch.softmax(logits_val, dim=-1)[:,1] #Get softmax score for blue

        # Predict on test set
        truth_test, logits_test = predict_classifier(model, test_dataset, 2, 100, device)
        preds_test = torch.argmax(logits_test, dim=-1).flatten()
        test_df[f"Prediction_{i}"] = preds_test
        test_df[f"Confidence_{i}"] = torch.softmax(logits_test, dim=-1)[:,1] #Get softmax score for blue

        # Predict for grid
        truth_grid, logits_grid = predict_classifier(model, grid_dataset, 2, 100, device)
        preds_grid = torch.argmax(logits_grid, dim=-1).flatten()
        grid_df[f"Prediction_{i}"] = preds_grid
        grid_df[f"Confidence_{i}"] = torch.softmax(logits_grid, dim=-1)[:,1] #Get softmax score for blue
    
        if biased_class < n_classes-1:
            biased_class = biased_class + 1
        else:
            biased_class = 0
    end = timer()
    print("Training time: ", timedelta(seconds=end-start))
    return val_df, test_df, grid_df

n_ensemble = 10
val_ensembles = [0]*len(n_data)
test_ensembles = [0]*len(n_data)
grid_ensembles = [0]*len(n_data)

for i in range(len(n_data)):
    logloss_min = 1
    for j in tqdm(range(20)):
        val_df, test_df, grid_df = train_ensemble(n_ensemble, n_data[i], bs_list[i], 2)
        val_df["Confidence_avg"] = val_df[[f"Confidence_{i}" for i in range(n_ensemble)]].mean(axis=1)
        val_df["Confidence_std"] = val_df[[f"Confidence_{i}" for i in range(n_ensemble)]].std(axis=1)
        val_df["Prediction_ensemble"] = 0
        mask = val_df["Confidence_avg"] > 0.5 # Equivalent to argmax for binary classification
        val_df.loc[mask, "Prediction_ensemble"] = 1

        ll = log_loss(val_df["class"], val_df["Confidence_avg"])
        preds = torch.Tensor(val_df["Confidence_avg"])
        target = torch.Tensor(val_df["class"])
        bce_l1 = BinaryCalibrationError(n_bins=15, norm='l1')
        ece = bce_l1(preds, target).item()
        print(f"n_train = {n_data[i]}, logloss={ll}, ECE= {ece}")

    if ll < logloss_min:
        print(f"New best values: n_train = {n_data[i]}, logloss={ll}, ECE= {ece}")
        logloss_min = ll

        val_ensembles[i] = val_df
        test_ensembles[i] = test_df
        grid_ensembles[i] = grid_df

        test_ensembles[i]["Confidence_avg"] = test_ensembles[i][[f"Confidence_{i}" for i in range(n_ensemble)]].mean(axis=1)
        test_ensembles[i]["Confidence_std"] = test_ensembles[i][[f"Confidence_{i}" for i in range(n_ensemble)]].std(axis=1)
        test_ensembles[i]["Prediction_ensemble"] = 0
        mask = test_ensembles[i]["Confidence_avg"] > 0.5
        test_ensembles[i].loc[mask, "Prediction_ensemble"] = 1

        grid_ensembles[i]["Confidence_avg"] = grid_ensembles[i][[f"Confidence_{i}" for i in range(n_ensemble)]].mean(axis=1)
        grid_ensembles[i]["Confidence_std"] = grid_ensembles[i][[f"Confidence_{i}" for i in range(n_ensemble)]].std(axis=1)
        grid_ensembles[i]["Prediction_ensemble"] = 0
        mask = grid_ensembles[i]["Confidence_avg"] > 0.5
        grid_ensembles[i].loc[mask, "Prediction_ensemble"] = 1
        # Save best prediction
    # Save best prediction
    if (not os.path.isdir(f"predictions/{trainfile}") ):
        os.mkdir(f"predictions/{trainfile}")
    if (not os.path.isdir(f"predictions/{trainfile}/CL") ):
        os.mkdir(f"predictions/{trainfile}/CL")
    #val_ensembles.to_csv(f"predictions/{trainfile}/{valfile}_SequentialNet_{n_ensemble}ensembles_ndata-{n_data[i]}.csv")
    test_ensembles[i].to_csv(f"predictions/{trainfile}/CL/{testfile}_SequentialNet_CL_{n_ensemble}ensembles_ndata-{n_data[i]}.csv")
    grid_ensembles[i].to_csv(f"predictions/{trainfile}/CL/grid_{tag}_SequentialNet_CL_{n_ensemble}ensembles_ndata-{n_data[i]}.csv")



import numpy as np
from  matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.metrics import roc_auc_score as roc_auc_score

from scipy.special import kl_div as kl_div
from sklearn.metrics import log_loss as log_loss

# import custom functions from src folder
from BayesianSequentialNet import BayesianSequentialNet
from machine_learning import *
from util import *

# Set up device
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device {torch.cuda.get_device_name(0)}")

# Machine learning options
x1_key = "x1"
x2_key = "x2"
nodes = 100
hidden_layers = 2
n_data = [250, 5000, 10000]
lr_list = np.linspace(0.001, 0.01, 20)
bs_list = [128, 1024, 1024]
epochs_list = [250, 250, 250]
patience_list = [100, 30, 30]

#Data constants
R2 = 3
k_red = 7
k_blue = 3
R1_min = 6
scale = 1
vary_a1 = False
vary_R2 = False
p_red = 0.5
polar = False
tag = f'r2_{R2}_kr{k_red}_kb{k_blue}_r1min{R1_min}_s{scale}_vary_r2_{vary_R2}_vary_a1_{vary_a1}_pRed_{p_red}'

# Read files
train_n = 50000
trainfile = f"train_n_{train_n}_{tag}"
valfile = f"val_n_5000_{tag}"
testfile = f"test_n_10000_{tag}"
truthfile = f"analytical_solution_x1_x2_grid_{tag}_nr1MC_4000"
truthfile_test = "analytical_solution_test_n_10000_r2_3_kr7_kb3_r1min6_s1_vary_r2_False_vary_a1_False_pRed_0.5_nr1MC_8000"
gridfile = f"x1_x2_grid"

train_data = pd.read_csv(f"../data/{trainfile}.csv")
val_data = pd.read_csv(f"../data/{valfile}.csv")
test_data = pd.read_csv(f"../data/{testfile}.csv")
truth_data = pd.read_csv(f"../analytical/results/{truthfile}.csv")
truth_data = truth_data[truth_data["r_x"] > R1_min-R2] # Remove undefined area
truth_test_data = pd.read_csv(f"../analytical/results/{truthfile_test}.csv")
grid_data = pd.read_csv(f"../data/{gridfile}.csv")

#Correct analytical solution if class distribution is not equal
if p_red != 0.5:
    truth_data["P_red_and_x"] = truth_data["P_red_and_x"]*(p_red)/0.5
    truth_data["P_blue_and_x"] = truth_data["P_blue_and_x"]*(1-p_red)/0.5
    truth_data["P_x"] = truth_data["P_red_and_x"] + truth_data["P_blue_and_x"]
    truth_data["P_red_given_x"] = truth_data["P_red_and_x"]/truth_data["P_x"]
    truth_data["P_blue_given_x"] = truth_data["P_blue_and_x"]/truth_data["P_x"]

# Prepare data for pytorch
train_data = word_to_int(train_data)
val_data = word_to_int(val_data)
test_data = word_to_int(test_data)

X_train = torch.Tensor(np.dstack((train_data[x1_key], train_data[x2_key]))).to(torch.float32)[0]
Y_train = label_maker(train_data["class"], 2)

X_val = torch.Tensor(np.dstack((val_data[x1_key], val_data[x2_key]))).to(torch.float32)[0]
Y_val = label_maker(val_data["class"], 2)

X_test = torch.Tensor(np.dstack((test_data[x1_key], test_data[x2_key]))).to(torch.float32)[0]
Y_test = torch.zeros(X_test.shape)

X_grid = torch.Tensor(np.dstack((grid_data[x1_key], grid_data[x2_key]))).to(torch.float32)[0]
Y_grid = torch.zeros(X_grid.shape)


# Create datasets for pytorch
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
grid_dataset = torch.utils.data.TensorDataset(X_grid, Y_grid)

###
test_dfs = [0]*len(n_data)
grid_dfs = [0]*len(n_data)
best_lr = [0]*len(n_data)

for i in range(len(n_data)):
    logloss_min = 1 #Minimize this
    n_train = n_data[i]
    epochs = epochs_list[i]
    batchsize = bs_list[i]
    patience = patience_list[i]
    for lr in lr_list:
        for j in range(20):
            val_df = pd.read_csv(f"../data/{valfile}.csv")
            test_df = pd.read_csv(f"../data/{testfile}.csv")
            grid_df = pd.read_csv(f"../data/{gridfile}.csv")

            model = BayesianSequentialNet(L=nodes, n_hidden=hidden_layers, activation="relu", in_channels=2, out_channels=2)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            

            train_dataset_small = torch.utils.data.TensorDataset(X_train[0:n_train], Y_train[0:n_train])
            train_results = train_bnn_classifier(model, train_dataset_small, val_dataset, batchsize, 
                                                epochs, device, optimizer, early_stopping=patience)

            val_df = predict_bnn(model, val_dataset, val_df, device, n_samples=20)
            ll = log_loss(val_df["class"], val_df["Est_prob_blue"])

            if ll < logloss_min:
                print(f"New best values: n_train = {n_train}, lr={lr}, logloss={ll}")
                logloss_min = ll
                best_lr[i] = lr
                test_dfs[i] = predict_bnn(model, test_dataset, test_df, device, n_samples=100)
                grid_dfs[i] = predict_bnn(model, grid_dataset, grid_df, device, n_samples=100)

        # Save prediction
        if (not os.path.isdir(f"predictions/{trainfile}") ):
            os.mkdir(f"predictions/scriptrun_{trainfile}")
        test_dfs[i].to_csv(f"predictions/{trainfile}/{testfile}_predicted_BNN_ndata-{n_data[i]}.csv")
        grid_dfs[i].to_csv(f"predictions/{trainfile}/grid_{tag}_predicted_BNN_ndata-{n_data[i]}.csv")


scores = calculate_metrics(test_dfs, grid_dfs, n_data, truth_data, truth_test_data, 
                           "Prediction", "Est_prob_blue", "Std_prob_blue")
scores["Best lr"] = best_lr
print(scores)

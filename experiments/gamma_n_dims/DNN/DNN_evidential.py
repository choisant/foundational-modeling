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
from evidential import train_evidential_classifier, predict_evidential_classifier
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
bs_list = [128, 128, 128, 128*2, 1024, 1024, 1024*2]
#n_data = [10000]
max_err_val = [0]*len(n_data)
for i in range(len(n_data)):
    if n_data[i] < 1000:
        max_err_val[i] = 0.15
    else:
        max_err_val[i] = 0.1
patience = 20

#Data constants
shapes = [2, 6]
scales = [5, 3]
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


epochs = 100 #Affects annealation coefficient
lr = 0.001

test_dfs = [0]*len(n_data)
grid_dfs = [0]*len(n_data)

for i in range(len(n_data)):

    logloss_min = 1
    test_dfs[i] = pd.read_csv(f"../data/{testfile}.csv")
    grid_dfs[i] = pd.read_csv(f"../data/{gridfile}.csv")
    for j in tqdm(range(20)):
        val_df = pd.read_csv(f"../data/{valfile}.csv")
        n_train = n_data[i]
        batchsize = bs_list[i]

        model = SequentialNet(L=200, n_hidden=3, activation="relu", in_channels=2, out_channels=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_dataset = torch.utils.data.TensorDataset(X_train[0:n_train], Y_train[0:n_train])
        training_results = train_evidential_classifier(model, train_dataset, 
                                val_dataset, batchsize=batchsize, epochs = epochs, 
                                device = device, optimizer = optimizer, early_stopping=patience)
        
        probs_val, uncertainties_val, beliefs_val = predict_evidential_classifier(model, val_dataset, 2, 100, device)
        preds_val = torch.argmax(probs_val, dim=-1).flatten()
        val_df["Prediction"] = preds_val
        val_df["Est_prob_blue"] = probs_val[:,1] #Get probability score for blue

        ll = log_loss(val_df["class"], val_df["Est_prob_blue"])
        preds = torch.Tensor(val_df["Est_prob_blue"])
        target = torch.Tensor(val_df["class"])
        bce_l1 = BinaryCalibrationError(n_bins=15, norm='l1')
        ece = bce_l1(preds, target).item()
        print(f"n_train = {n_data[i]}, logloss={ll}, ECE= {ece}, best value: {logloss_min}")

        if ll < logloss_min:
            print(f"New best values: n_train = {n_data[i]}, logloss={ll}, ECE= {ece}")
            logloss_min = ll

            probs_test, uncertainties_test, beliefs_test = predict_evidential_classifier(model, test_dataset, 2, 100, device)
            preds_test = torch.argmax(probs_test, dim=-1).flatten()
            test_dfs[i]["Prediction"] = preds_test
            test_dfs[i]["Est_prob_blue"] = probs_test[:,1] #Get probability score for blue
            test_dfs[i]["Std_prob_blue"] = uncertainties_test[:,1]
            test_dfs[i]["Beliefs"] = beliefs_test[:,1]

            probs_grid, uncertainties_grid, beliefs_grid = predict_evidential_classifier(model, grid_dataset, 2, 100, device)
            preds_grid = torch.argmax(probs_grid, dim=-1).flatten()
            grid_dfs[i]["Prediction"] = preds_grid
            grid_dfs[i]["Est_prob_blue"] = probs_grid[:,1] #Get probability score for blue
            grid_dfs[i]["Std_prob_blue"] = uncertainties_grid[:,1]
            grid_dfs[i]["Beliefs"] = beliefs_grid[:,1]

        # Save best prediction
    if (not os.path.isdir(f"predictions/{trainfile}") ):
        os.mkdir(f"predictions/{trainfile}")
    if (not os.path.isdir(f"predictions/{trainfile}/evidential") ):
        os.mkdir(f"predictions/{trainfile}/evidential")
    test_dfs[i].to_csv(f"predictions/{trainfile}/evidential/{testfile}_SequentialNet_evidential_best_ndata-{n_data[i]}.csv")
    grid_dfs[i].to_csv(f"predictions/{trainfile}/evidential/grid_{tag}_SequentialNet_evidential_best_ndata-{n_data[i]}.csv")

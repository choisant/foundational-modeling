import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True
pd.option_context('mode.chained_assignment','raise')
import sys
import os
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.optim as optim
from torchmetrics.classification import BinaryCalibrationError

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score as roc_auc_score
from scipy.special import kl_div as kl_div
from sklearn.metrics import log_loss as log_loss

# import custom functions from src folder
module_path = str(Path.cwd() / "../../../src")

if module_path not in sys.path:
    sys.path.append(module_path)

from SequentialNet import SequentialNet
from machine_learning import *
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

# Machine learning option
ALGORITHM_NAME = "CL"
VARY_HYPERPARAMS = True # Increases run time substantially and does not save any predictions/models
x1_key = "x1"
x2_key = "x2"
n_data = [250, 500, 1000, 2000, 3000, 5000, 10000]
bs_list = [128, 128, 128, 128*2, 1024, 1024, 1024*2] #Batchsize

# SGD options
if VARY_HYPERPARAMS == False:
    n_runs = 20
    lr = 0.001
    weight_decay = 0.01
    n_nodes = 200
    n_layers = 3
    bias_weight = 0.1
    SAVE_PREDS = True
else:
    n_runs = 1
    SAVE_PREDS = False #Don't save predictions for hyperparam search mode
    hyperparams = {
        "lr" : [0.01, 0.001, 0.0001],
        "weight_decay" : [0.1, 0.01, 0.001],
        "layers" : [1, 3, 8],
        "bias_weight" : [0.5, 0.1, 0.2]
    }
    n_data = [250, 1000, 5000]
    bs_list = [128, 256, 1024]

n_ensemble = 20
n_classes = 2
patience = 20 # For early stopping
epochs = 250

# Data constants
shapes = [2, 4]
scales = [4, 3]
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
large_gridfile = f"grid_r_a1_2500_{tag}"

# Process data
train_data = pd.read_csv(f"../data/{trainfile}.csv")
val_data = pd.read_csv(f"../data/{valfile}.csv")
test_data = pd.read_csv(f"../data/{testfile}.csv")
grid_data = pd.read_csv(f"../data/{gridfile}.csv")
grid_rmax = grid_data["x1"].max()
large_grid_data = pd.read_csv(f"../data/{large_gridfile}.csv")
large_grid_rmax = large_grid_data["x1"].max()

X_train = torch.Tensor(np.dstack((train_data[x1_key], train_data[x2_key]))).to(torch.float32)[0]
Y_train = label_maker(train_data["class"], 2)
X_val = torch.Tensor(np.dstack((val_data[x1_key], val_data[x2_key]))).to(torch.float32)[0]
Y_val = label_maker(val_data["class"], 2)
X_test = torch.Tensor(np.dstack((test_data[x1_key], test_data[x2_key]))).to(torch.float32)[0]
Y_test = label_maker(test_data["class"], 2)
X_grid = torch.Tensor(np.dstack((grid_data[x1_key], grid_data[x2_key]))).to(torch.float32)[0]
Y_grid = torch.zeros(X_grid.shape)
X_large_grid = torch.Tensor(np.dstack((large_grid_data[x1_key], large_grid_data[x2_key]))).to(torch.float32)[0]
Y_large_grid = torch.zeros(X_large_grid.shape)

# Create datasets for pytorch
val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
grid_dataset = torch.utils.data.TensorDataset(X_grid, Y_grid)
large_grid_dataset = torch.utils.data.TensorDataset(X_large_grid, Y_large_grid)

# Vary hyperparameters
def create_df(ntrain, hyperparam_dict):
    # Total number of combinations of hyperparams
    n_var_hyperparams = 1
    for key in hyperparams.keys():
        n_var_hyperparams = n_var_hyperparams*len(hyperparam_dict[key])

    # Create dataframe
    df = pd.DataFrame(columns=hyperparam_dict.keys())
    df["ntrain"] = np.array([ntrain]*n_var_hyperparams).flatten()
    # With all permutations
    iter_perms_down = n_var_hyperparams
    iter_perms_up = 1
    for key in hyperparam_dict.keys():
        values = hyperparam_dict[key]
        iter_perms_down = int(iter_perms_down/len(values))
        df[key] = np.array([[x]*iter_perms_down for x in values]*iter_perms_up).flatten()
        iter_perms_up = int(iter_perms_up*len(values))

    # Nodes
    df["nodes"] = np.zeros(len(df))
    nodes = [20, 200, 2000]
    for i in range(len(hyperparam_dict["layers"])):
        n_layers = hyperparam_dict["layers"][i]
        df_copy = df.copy()
        mask = df_copy["layers"] == n_layers
        df.loc[mask, "nodes"] = nodes[i]
    return df


def train_ensemble(n_ensemble:int, n_train:int, batchsize:int, n_nodes:int, n_layers:int, lr, weight_decay, n_classes:int, bias_weight):
    # Stupid function with global and local variables
    if (n_ensemble%n_classes) != 0:
        print("Please set n_ensembles to n_classes*int.")
        return None
    else:
        biased_class = 0
    
    val_df = pd.read_csv(f"../data/{valfile}.csv")
    test_df = pd.read_csv(f"../data/{testfile}.csv")
    grid_df = pd.read_csv(f"../data/{gridfile}.csv")
    large_grid_df = pd.read_csv(f"../data/{large_gridfile}.csv")
    # Timer
    start_train = timer()
    print(f"Starting training of {n_ensemble} ensembles with {n_train} training points.")
    for i in range(n_ensemble):
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train[0:n_train], Y_train[0:n_train])

        # Create new model
        model = SequentialNet(L=n_nodes, n_hidden=n_layers, activation="relu", in_channels=2, out_channels=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train model
        training_results = train_classifier(model, train_dataset, 
                                val_dataset, batchsize=batchsize, epochs = epochs, 
                                device = device, optimizer = optimizer, early_stopping=patience,
                                biased_class=biased_class, bias_weight=bias_weight)
        
        # Predict on validation set
        truth_val, logits_val = predict_classifier(model, val_dataset, 2, 100, device)
        preds_val = torch.argmax(logits_val, dim=-1).flatten()
        val_df[f"Prediction_{i}"] = preds_val
        val_df[f"Est_prob_c1_{i}"] = torch.softmax(logits_val, dim=-1)[:,1] #Get softmax score for blue

        # Predict on test set
        truth_test, logits_test = predict_classifier(model, test_dataset, 2, 100, device)
        preds_test = torch.argmax(logits_test, dim=-1).flatten()
        test_df[f"Prediction_{i}"] = preds_test
        test_df[f"Est_prob_c1_{i}"] = torch.softmax(logits_test, dim=-1)[:,1] #Get softmax score for blue

        # Predict for grid
        truth_grid, logits_grid = predict_classifier(model, grid_dataset, 2, 100, device)
        preds_grid = torch.argmax(logits_grid, dim=-1).flatten()
        grid_df[f"Prediction_{i}"] = preds_grid
        grid_df[f"Est_prob_c1_{i}"] = torch.softmax(logits_grid, dim=-1)[:,1] #Get softmax score for blue

        # Predict for large grid
        truth_large_grid, logits_large_grid = predict_classifier(model, large_grid_dataset, 2, 100, device)
        preds_large_grid = torch.argmax(logits_large_grid, dim=-1).flatten()
        large_grid_df[f"Prediction_{i}"] = preds_large_grid
        large_grid_df[f"Est_prob_c1_{i}"] = torch.softmax(logits_large_grid, dim=-1)[:,1] #Get softmax score for blue

        if biased_class < n_classes-1:
            biased_class = biased_class + 1
        else:
            biased_class = 0

    end_train = timer()
    total_time = timedelta(seconds=end_train-start_train)
    print("Training time: ", timedelta(seconds=end_train-start_train))
    return val_df, test_df, grid_df, large_grid_df, total_time


"""
Start training
"""

if VARY_HYPERPARAMS == False:
    val_ensembles = [0]*len(n_data)
    test_ensembles = [0]*len(n_data)
    grid_ensembles = [0]*len(n_data)
    large_grid_ensembles = [0]*len(n_data)

    for i in range(len(n_data)):
        logloss_min = 1
        for j in tqdm(range(n_runs)):
            val_df, test_df, grid_df, large_grid_df, total_time = train_ensemble(n_ensemble, n_data[i], bs_list[i])
            val_df["Est_prob_c1"] = val_df[[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].mean(axis=1)
            val_df["Std_prob_c1"] = val_df[[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].std(axis=1)
            val_df["Prediction"] = 0
            mask = val_df["Est_prob_c1"] > 0.5 # Equivalent to argmax for binary classification
            val_df.loc[mask, "Prediction"] = 1

            ll = log_loss(val_df["class"], val_df["Est_prob_c1"])
            print(f"n_train = {n_data[i]}, logloss={ll}")

            if ll < logloss_min:
                print(f"New best values: n_train = {n_data[i]}, logloss={ll}")
                logloss_min = ll

                val_ensembles[i] = val_df
                test_ensembles[i] = test_df
                grid_ensembles[i] = grid_df
                large_grid_ensembles[i] = large_grid_df

                test_ensembles[i]["Est_prob_c1"] = test_ensembles[i][[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].mean(axis=1)
                test_ensembles[i]["Std_prob_c1"] = test_ensembles[i][[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].std(axis=1)
                test_ensembles[i]["Prediction"] = 0
                mask = test_ensembles[i]["Est_prob_c1"] > 0.5
                test_ensembles[i].loc[mask, "Prediction"] = 1

                grid_ensembles[i]["Est_prob_c1"] = grid_ensembles[i][[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].mean(axis=1)
                grid_ensembles[i]["Std_prob_c1"] = grid_ensembles[i][[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].std(axis=1)
                grid_ensembles[i]["Prediction"] = 0
                mask = grid_ensembles[i]["Est_prob_c1"] > 0.5
                grid_ensembles[i].loc[mask, "Prediction"] = 1

                large_grid_ensembles[i]["Est_prob_c1"] = large_grid_ensembles[i][[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].mean(axis=1)
                large_grid_ensembles[i]["Std_prob_c1"] = large_grid_ensembles[i][[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].std(axis=1)
                large_grid_ensembles[i]["Prediction"] = 0
                mask = large_grid_ensembles[i]["Est_prob_c1"] > 0.5
                large_grid_ensembles[i].loc[mask, "Prediction"] = 1
        
        # Save best prediction
        if (not os.path.isdir(f"predictions/{trainfile}") ):
            os.mkdir(f"predictions/{trainfile}")
        if (not os.path.isdir(f"predictions/{trainfile}/{ALGORITHM_NAME}") ):
            os.mkdir(f"predictions/{trainfile}/{ALGORITHM_NAME}")
        test_ensembles[i].to_csv(f"predictions/{trainfile}/{ALGORITHM_NAME}/{testfile}_ndata-{n_data[i]}.csv")
        grid_ensembles[i].to_csv(f"predictions/{trainfile}/{ALGORITHM_NAME}/grid_{tag}_ndata-{n_data[i]}.csv")
        large_grid_ensembles[i].to_csv(f"predictions/{trainfile}/{ALGORITHM_NAME}/large_grid_{tag}_ndata-{n_data[i]}.csv")


else:
    print("Starting hyperparameter search")
    start = timer()
    errors=0
    # Run more times for statistics
    for j in range(n_runs):
        # Run once per dataset per run
        for k in range(len(n_data)):
            n_train = n_data[k]
            batchsize = bs_list[k]
            train_dataset = torch.utils.data.TensorDataset(X_train[0:n_train], Y_train[0:n_train])
            df_run = create_df(n_train, hyperparams)
            metric_keys = ["ACC", "LogLoss", "Mean KL-div", "ECE", "WD", "Mean UQ", 
                        "Std UQ", "Min UQ", "Max UQ",
                        "Mean Pc1 OOD", "Std Pc1 OOD", "Max Pc1 OOD", "Min Pc1 OOD",
                        "Mean UQ OOD", "Std UQ OOD", "Max UQ OOD", "Min UQ OOD"]
            for key in metric_keys:
                df_run[key] = None
            print(f"Testing {len(df_run)} hyperparameter combinations in run nr {j + 1} out of {n_runs}")
            # Run once for each hyperparameter combination
            i = 0
            while i < len(df_run):
                val_df = pd.read_csv(f"../data/{valfile}.csv")
                large_grid_df = pd.read_csv(f"../data/{large_gridfile}.csv")
                
                # Get parameters
                lr = df_run["lr"].values[i]
                weight_decay = df_run["weight_decay"].values[i]
                n_layers = int(df_run["layers"].values[i])
                n_nodes = int(df_run["nodes"].values[i])
                bias_weight = df_run["bias_weight"].values[i]

                # Train model
                val_df, test_df, grid_df, large_grid_df, total_time = train_ensemble(n_ensemble, n_train, batchsize, 
                                                                                    n_nodes, n_layers, lr, weight_decay,
                                                                                    n_classes=n_classes, bias_weight=bias_weight)
                
                # Evaluate on validation set
                val_df["Est_prob_c1"] = val_df[[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].mean(axis=1)
                val_df["Std_prob_c1"] = val_df[[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].std(axis=1)
                val_df["Prediction"] = 0
                mask = val_df["Est_prob_c1"] > 0.5 # Equivalent to argmax for binary classification
                val_df.loc[mask, "Prediction"] = 1

                len_nan = len(val_df[val_df.isnull().any(axis=1)])
                if (len_nan < len(val_df)):
                    if len_nan > 0:
                        print(f"Dropping {len_nan} rows of NaNs from validation file")
                        val_df = val_df.dropna()
                    # Make them tensors (might be redundant)
                    est_probs = torch.Tensor(val_df["Est_prob_c1"])
                    pred_class = torch.Tensor(val_df["Prediction"])
                    target = torch.Tensor(val_df["class"])
                    truth_probs = torch.Tensor(val_df["p_c1_given_r"])

                    # Calculate metrics
                    ll = log_loss(target, est_probs)
                    df_run.loc[i, "ACC"] = accuracy_score(target, pred_class)
                    df_run.loc[i, "LogLoss"] = ll
                    bce_l1 = BinaryCalibrationError(n_bins=15, norm='l1')
                    ece = bce_l1(est_probs, target).item()
                    df_run.loc[i, "ECE"] = ece
                    df_run.loc[i, "WD"] = wasserstein_distance(truth_probs, est_probs)
                    df_run.loc[i, "Mean KL-div"] = kl_div(truth_probs, est_probs).mean().numpy()
                    df_run.loc[i, "Mean UQ"] = val_df["Std_prob_c1"].mean()
                    df_run.loc[i, "Std UQ"] = val_df["Std_prob_c1"].std()
                    df_run.loc[i, "Max UQ"] = val_df["Std_prob_c1"].max()
                    df_run.loc[i, "Min UQ"] = val_df["Std_prob_c1"].min()

                    # Evaluate on large grid
                    large_grid_df["Est_prob_c1"] = large_grid_df[[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].mean(axis=1)
                    large_grid_df["Std_prob_c1"] = large_grid_df[[f"Est_prob_c1_{i}" for i in range(n_ensemble)]].std(axis=1)
                    large_grid_df["Prediction_ensemble"] = 0
                    mask = large_grid_df["Est_prob_c1"] > 0.5
                    large_grid_df.loc[mask, "Prediction_ensemble"] = 1
                    large_r_df = large_grid_df.copy()[large_grid_df["r"] > 700]

                    df_run.loc[i, "Mean Pc1 OOD"] = large_r_df["Est_prob_c1"].mean()
                    df_run.loc[i, "Std Pc1 OOD"] = large_r_df["Est_prob_c1"].std()
                    df_run.loc[i, "Max Pc1 OOD"] = large_r_df["Est_prob_c1"].max()
                    df_run.loc[i, "Min Pc1 OOD"] = large_r_df["Est_prob_c1"].min()

                    df_run.loc[i, "Mean UQ OOD"] = large_r_df["Std_prob_c1"].mean()
                    df_run.loc[i, "Std UQ OOD"] = large_r_df["Std_prob_c1"].std()
                    df_run.loc[i, "Max UQ OOD"] = large_r_df["Std_prob_c1"].max()
                    df_run.loc[i, "Min UQ OOD"] = large_r_df["Std_prob_c1"].min()

                    df_run.loc[i, "Training time"] = total_time

                    # Save for every line, in case something goes wrong
                    if (not os.path.isdir(f"gridsearch") ):
                        os.mkdir(f"gridsearch")
                    if (not os.path.isdir(f"gridsearch/{trainfile}") ):
                        os.mkdir(f"gridsearch/{trainfile}")
                    if (not os.path.isdir(f"gridsearch/{trainfile}/{ALGORITHM_NAME}") ):
                        os.mkdir(f"gridsearch/{trainfile}/{ALGORITHM_NAME}")
                    df_run.to_csv(f"gridsearch/{trainfile}/{ALGORITHM_NAME}/results_run{j}_ntrain_{n_train}.csv")
                    i = i + 1
                else:
                    print("WTF is going on here?? Skipping these")
                    print(df_run.loc[i])
                    print(val_df["Est_prob_c1"])
                    errors = errors + 1

    end = timer()
    print("Finsihed conflictual loss grid search")
    print("Grid search time: ", timedelta(seconds=end-start))
    print("Total nr of errors caught: ", errors)


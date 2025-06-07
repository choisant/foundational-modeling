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
import argparse

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
from evidential import train_evidential_classifier, predict_evidential_classifier
from util import label_maker

##Parser
parser = argparse.ArgumentParser()
parser.add_argument('--shape1', type=int, required=True, help="Integer. Shape factor of class 1.")
parser.add_argument('--shape2', type=int, required=True, help="Integer. Shape factor of class 2.")
parser.add_argument('--scale1', type=int, required=True, help="Integer. Scale factor of class 1.")
parser.add_argument('--scale2', type=int, required=True, help="Integer. Scale factor of class 2.")
parser.add_argument('-g', '--grid', action='store_true', help="Turn on hyperparameter grid search mode. Default off.")
args = parser.parse_args()

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
ALGORITHM_NAME = "evidential"
VARY_HYPERPARAMS = args.grid # Increases run time substantially and does not save any predictions/models
x1_key = "x1"
x2_key = "x2"

# SGD options
if VARY_HYPERPARAMS == False:
    n_data = [250, 500, 1000, 2000, 3000, 5000, 10000]
    bs_list = [128, 128, 256, 256, 1024, 1024, 1024*2] #Batchsize
    n_runs = 20
    lr = 0.0001
    weight_decay = 0.01
    annealing_coef = 0.02 #Constant multiplier for KL div term
    n_nodes = 2000
    n_layers = 8
    SAVE_PREDS = True
else:
    n_runs = 20
    SAVE_PREDS = False #Don't save predictions for hyperparam search mode
    hyperparams = {
        "lr" : [0.01, 0.001, 0.0001],
        "weight_decay" : [0.1, 0.01, 0.001],
        "annealing_coef" : [0.0002, 0.002, 0.02],
        "layers" : [1, 3, 8]
    }
    n_data = [250, 1000, 5000]
    bs_list = [128, 256, 1024]

patience = 20 # For early stopping
epochs = 200

# Data constants
shapes = [args.shape1, args.shape2]
scales = [args.scale1, args.scale2]
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

"""
Start training
"""

if VARY_HYPERPARAMS == False:
    # Create folders
    if (not os.path.isdir(f"predictions/{trainfile}") ):
            os.mkdir(f"predictions/{trainfile}")
    if (not os.path.isdir(f"predictions/{trainfile}/{ALGORITHM_NAME}") ):
        os.mkdir(f"predictions/{trainfile}/{ALGORITHM_NAME}")
        
    print(f"Starting training of {n_runs} models for increasing nr of datapoints: {n_data} for study.")
    test_dfs = [0]*len(n_data)
    grid_dfs = [0]*len(n_data)
    large_grid_dfs = [0]*len(n_data)

    for i in range(len(n_data)):
        logloss_min = 1
        test_dfs[i] = pd.read_csv(f"../data/{testfile}.csv")
        grid_dfs[i] = pd.read_csv(f"../data/{gridfile}.csv")
        large_grid_dfs[i] = pd.read_csv(f"../data/{large_gridfile}.csv")
        for j in tqdm(range(n_runs)):
            val_df = pd.read_csv(f"../data/{valfile}.csv")
            n_train = n_data[i]
            batchsize = bs_list[i]

            model = SequentialNet(L=n_nodes, n_hidden=n_layers, activation="relu", in_channels=2, out_channels=2).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            train_dataset = torch.utils.data.TensorDataset(X_train[0:n_train], Y_train[0:n_train])
            training_results = train_evidential_classifier(model, train_dataset, 
                                    val_dataset, batchsize=batchsize, epochs = epochs, 
                                    device = device, optimizer = optimizer, 
                                    annealing_coef=annealing_coef, early_stopping=patience)
            
            probs_val, uncertainties_val, beliefs_val = predict_evidential_classifier(model, val_dataset, 2, 100, device)
            preds_val = torch.argmax(probs_val, dim=-1).flatten()
            val_df["Prediction"] = preds_val
            val_df["Est_prob_c1"] = probs_val[:,1] #Get probability score for blue

            ll = log_loss(val_df["class"], val_df["Est_prob_c1"])
            preds = torch.Tensor(val_df["Est_prob_c1"])
            target = torch.Tensor(val_df["class"])
            bce_l1 = BinaryCalibrationError(n_bins=15, norm='l1')
            ece = bce_l1(preds, target).item()
            #print(f"n_train = {n_data[i]}, logloss={ll}, ECE= {ece}, best value: {logloss_min}")

            if ll < logloss_min:
                #print(f"New best values: n_train = {n_data[i]}, logloss={ll}, ECE= {ece}")
                logloss_min = ll

                test_dfs[i] = pd.read_csv(f"../data/{testfile}.csv")
                grid_dfs[i] = pd.read_csv(f"../data/{gridfile}.csv")
                large_grid_dfs[i] = pd.read_csv(f"../data/{large_gridfile}.csv")

                probs_test, uncertainties_test, beliefs_test = predict_evidential_classifier(model, test_dataset, 2, 100, device)
                preds_test = torch.argmax(probs_test, dim=-1).flatten()
                test_dfs[i]["Prediction"] = preds_test
                test_dfs[i]["Est_prob_c1"] = probs_test[:,1] #Get probability score for blue
                test_dfs[i]["Std_prob_c1"] = uncertainties_test[:,1]
                test_dfs[i]["Beliefs"] = beliefs_test[:,1]

                probs_grid, uncertainties_grid, beliefs_grid = predict_evidential_classifier(model, grid_dataset, 2, 100, device)
                preds_grid = torch.argmax(probs_grid, dim=-1).flatten()
                grid_dfs[i]["Prediction"] = preds_grid
                grid_dfs[i]["Est_prob_c1"] = probs_grid[:,1] #Get probability score for blue
                grid_dfs[i]["Std_prob_c1"] = uncertainties_grid[:,1]
                grid_dfs[i]["Beliefs"] = beliefs_grid[:,1]

                probs_large_grid, uncertainties_large_grid, beliefs_large_grid = predict_evidential_classifier(model, large_grid_dataset, 2, 100, device)
                preds_large_grid = torch.argmax(probs_large_grid, dim=-1).flatten()
                large_grid_dfs[i]["Prediction"] = preds_large_grid
                large_grid_dfs[i]["Est_prob_c1"] = probs_large_grid[:,1] #Get probability score for blue
                large_grid_dfs[i]["Std_prob_c1"] = uncertainties_large_grid[:,1]
                large_grid_dfs[i]["Beliefs"] = beliefs_large_grid[:,1]

                # Save best prediction
                test_dfs[i].to_csv(f"predictions/{trainfile}/{ALGORITHM_NAME}/{testfile}_ndata-{n_data[i]}.csv")
                grid_dfs[i].to_csv(f"predictions/{trainfile}/{ALGORITHM_NAME}/grid_{tag}_ndata-{n_data[i]}.csv")
                large_grid_dfs[i].to_csv(f"predictions/{trainfile}/{ALGORITHM_NAME}/large_grid_{tag}_ndata-{n_data[i]}.csv")


else:
    print("Starting hyperparameter search")
    start = timer()
    errors = 0
    # Run more times for statistics
    for j in range(n_runs):
        # Run once per dataset per run
        for k in range(len(n_data)):
            n_train = n_data[k]
            batchsize = bs_list[k]
            df_run = create_df(n_train, hyperparams)
            train_dataset = torch.utils.data.TensorDataset(X_train[0:n_train], Y_train[0:n_train])
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
                annealing_coef = df_run["annealing_coef"].values[i]
                
                # Initialize model and optimizer
                model = SequentialNet(L=n_nodes, n_hidden=n_layers, activation="relu", in_channels=2, out_channels=2).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                
                # Train model
                training_results = train_evidential_classifier(model, train_dataset, 
                                        val_dataset, batchsize=batchsize, epochs = epochs, 
                                        device = device, optimizer = optimizer, 
                                        annealing_coef=annealing_coef, early_stopping=patience)
                # Evaluate on validation set
                est_probs_val, uncertainties_val, beliefs_val = predict_evidential_classifier(model, val_dataset, 2, 100, device)
                preds_val = torch.argmax(est_probs_val, dim=-1).flatten()
                val_df["Prediction"] = preds_val
                val_df["Est_prob_c1"] = est_probs_val[:,1] #Get probability score for blue
                val_df["Std_prob_c1"] = uncertainties_val[:,1]
                
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
                    est_probs_large_grid, uncertainties_large_grid, beliefs_large_grid = predict_evidential_classifier(model, 
                        large_grid_dataset, 2, 100, device)
                    preds_large_grid = torch.argmax(est_probs_large_grid, dim=-1).flatten()
                    large_grid_df["Prediction"] = preds_large_grid
                    large_grid_df["Est_prob_c1"] = est_probs_large_grid[:,1] #Get probability score for blue
                    large_grid_df["Std_prob_c1"] = uncertainties_large_grid[:,1]

                    large_r_df = large_grid_df.copy()[large_grid_df["r"] > 700]

                    df_run.loc[i, "Mean Pc1 OOD"] = large_r_df["Est_prob_c1"].mean()
                    df_run.loc[i, "Std Pc1 OOD"] = large_r_df["Est_prob_c1"].std()
                    df_run.loc[i, "Max Pc1 OOD"] = large_r_df["Est_prob_c1"].max()
                    df_run.loc[i, "Min Pc1 OOD"] = large_r_df["Est_prob_c1"].min()

                    df_run.loc[i, "Mean UQ OOD"] = large_r_df["Std_prob_c1"].mean()
                    df_run.loc[i, "Std UQ OOD"] = large_r_df["Std_prob_c1"].std()
                    df_run.loc[i, "Max UQ OOD"] = large_r_df["Std_prob_c1"].max()
                    df_run.loc[i, "Min UQ OOD"] = large_r_df["Std_prob_c1"].min()

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
    end = timer()
    print("Finsihed evidential deep learning grid search")
    print("Grid search time: ", timedelta(seconds=end-start))
    print("Total nr of errors caught: ", errors)

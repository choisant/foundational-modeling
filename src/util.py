from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sn
import torch

from torchmetrics.classification import BinaryCalibrationError

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score as roc_auc_score
from scipy.special import kl_div as kl_div
from sklearn.metrics import log_loss as log_loss



def plot_conf_matrix(df, truthkey, predkey, labels, ax):
    """
    plot confusion matrix
    """
    # Get total accuracy
    accuracy = accuracy_score(df[truthkey], df[predkey], normalize=True)
    #Generate the confusion matrix
    cf_matrix = confusion_matrix(df[truthkey], df[predkey], normalize="true")
    cf_matrix = 100*np.round(cf_matrix, 4)
    group_percentages = ["{0:0.2%}".format(value/100) for value in cf_matrix.flatten()]
    annot = [f"{item}" for item in group_percentages]
    annot = np.asarray(annot).reshape(len(labels),len(labels))
    sn.heatmap(
            cf_matrix, 
            ax=ax, 
            annot=annot, 
            cmap='rocket', 
            linewidths=1.0, 
            linecolor='black',
            cbar = True, 
            square=True, 
            fmt='',
            cbar_kws={"format": "%.0f%%", "shrink": 1.0},
            vmin=0,
            vmax=100,
            annot_kws={"size": 14}
        )

    #ax.set_title('Confusion matrix\n\n', size=24)
    ax.set_xlabel('Predicted labels', size=14)
    ax.set_ylabel('True labels', size=14)

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(labels, size=12, rotation=20)
    ax.yaxis.set_ticklabels(labels, size=12, rotation=70)
    ax.tick_params(which="both", left=False, bottom=False, top=False, right=False)
    acc_rounded = "{0:0.2%}".format(accuracy)
    ax.set_title(f"Accuracy: {acc_rounded}", fontsize=14)
    return ax

###### Plots
def pink_black_green_cmap():
    #c = ["#b2182b","#ef8a62","#fddbc7", "#d9d9d9", "#d1e5f0","#67a9cf","#2166ac"]
    c = ["#fdb7e0","#000000","#d4f48d"]
    v = [0,.5,1.]
    l = list(zip(v,c))
    cmap = LinearSegmentedColormap.from_list('rg',l, N=256)
    return cmap

def red_blue_cmap():
    #c = ["#b2182b","#ef8a62","#fddbc7", "#d9d9d9", "#d1e5f0","#67a9cf","#2166ac"]
    c = ["#a50026","#d73027","#fdae61", "#f7f7f7", "#abd9e9","#4575b4","#313695"]
    v = [0,.2,.4,.5,0.6,.8,1.]
    l = list(zip(v,c))
    cmap = LinearSegmentedColormap.from_list('rg',l, N=256)
    return cmap



def plot_grid(grid_df, pred_key, ax, suptitle, nx:int = 100):
    ax.set_title(suptitle)
    x1_lim = 25
    x2_lim = 25

    cmap = red_blue_cmap()
    hue_norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False)

    ax.hist2d(x= grid_df["x1"], y= grid_df["x2"], weights=grid_df[pred_key], 
                bins = nx,
                norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False),
                cmap=cmap, rasterized=True, edgecolor='face')
    
    ax.set_xlim(-x1_lim, x1_lim)
    ax.set_ylim(-x2_lim, x2_lim)
    ax.set_xlabel(r"x$_1$", fontsize=14)
    ax.set_ylabel(r"x$_2$", fontsize=14)

    ax.tick_params(which="both", direction="inout", bottom=True, left=True, labelsize=14, pad=5, length=4, width=1)
    ax.tick_params(which="major", length=6)
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-20, 0, 20])
    ax.minorticks_on()
    ax.set_aspect('equal', adjustable='box')

    return ax

def plot_data(df, ax, suptitle, rmax:float=25.0):
    ax.set_title(suptitle)
    df_red = df[df["class"] == 0]
    sn.scatterplot(df_red, x="x1", y = "x2", c="red", alpha=0.5, ax=ax)
    
    df_blue = df[df["class"] == 1]
    sn.scatterplot(df_blue, x="x1", y = "x2", c="blue", alpha=0.5, ax=ax)
    
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_xlabel(r"x$_1$", fontsize=14)
    ax.set_ylabel(r"x$_2$", fontsize=14)

    ax.tick_params(which="both", direction="inout", bottom=True, left=True, labelsize=14, pad=5, length=4, width=1)
    ax.tick_params(which="major", length=6)
    ax.set_xticks([-rmax, 0, rmax])
    ax.set_yticks([-rmax, 0, rmax])
    ax.set_aspect('equal', adjustable='box')
    ax.minorticks_on()

    return ax

def plot_results(df, pred_key, ax, suptitle, error_key=None, grid=False, rmax:float=25.0, bins:int=100):
    ax.set_title(suptitle)
    cmap = red_blue_cmap()
    
    if grid:
        ax.hist2d(x= df["x1"], y=df["x2"], weights=df[pred_key], 
                bins = bins,
                norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False),
                cmap=cmap, rasterized=True, edgecolor='face')
    else:
    
        if error_key == None:
            sn.scatterplot(data = df, x="x1", y="x2", ax = ax, hue=pred_key, 
                        hue_norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False),
                            palette=cmap, legend=False)
        else:
            sn.scatterplot(data = df, x="x1", y="x2", ax = ax, hue=pred_key, 
                        size=error_key, size_norm = (0.1, 0.2), sizes=(10, 200),
                            hue_norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False),
                            palette=cmap, legend=False)
    
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    #ax.set_xlabel(r"x$_1$", fontsize=14)
    #ax.set_ylabel(r"x$_2$", fontsize=14)

    ax.tick_params(which="both", direction="inout", bottom=True, left=True, labelsize=14, pad=5, length=4, width=1)
    ax.tick_params(which="major", length=6)
    ax.set_xticks([-rmax, 0, rmax])
    ax.set_yticks([-rmax, 0, rmax])
    ax.set_aspect('equal', adjustable='box')
    ax.minorticks_on()
    return ax

def plot_diff(df_pred, df_truth, pred_key, truth_key, ax, suptitle, max_val=0.5, rmax:float=25.0, bins:int=100):
    #Assume my truthfile format
    ax.set_title(suptitle)
    #Absolute value
    df_pred["Diff_truth"] = df_pred[pred_key] - df_truth[truth_key]
    
    #sn.scatterplot(data = df_pred, x="x1", y="x2", ax = ax, hue="Diff_truth", 
    #                palette="dark:#5A9_r", legend=False, linewidth=0)
    ax.hist2d(x=df_pred["x1"], y=df_pred["x2"], weights=df_pred["Diff_truth"], 
                bins = bins,
                norm = mpl.colors.Normalize(vmin=-max_val, vmax=max_val, clip=False),
                cmap=pink_black_green_cmap(), rasterized=True, edgecolor='face'
                )
    
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    #ax.set_xlabel(r"x$_1$", fontsize=14)
    #ax.set_ylabel(r"x$_2$", fontsize=14)

    ax.tick_params(which="both", direction="inout", bottom=True, left=True, labelsize=14, pad=5, length=4, width=1)
    ax.tick_params(which="major", length=6)
    ax.set_xticks([-rmax, 0, rmax])
    ax.set_yticks([-rmax, 0, rmax])
    ax.set_aspect('equal', adjustable='box')
    ax.minorticks_on()
    return ax

def plot_std(df, pred_key, ax, suptitle, grid=False, max_val=0.5, rmax:float=25.0, bins:int=100):
    ax.set_title(suptitle)
    
    if grid:
        ax.hist2d(x= df["x1"], y=df["x2"], weights=df[pred_key], 
                bins = bins,
                norm = mpl.colors.Normalize(vmin=0, vmax=max_val, clip=False),
                cmap="inferno", rasterized=True, edgecolor='face')
    else:

        sn.scatterplot(data = df, x="x1", y="x2", ax = ax, hue=pred_key, 
                    hue_norm = mpl.colors.Normalize(vmin=0, vmax=max_val, clip=False),
                    palette="inferno",
                    legend=False, linewidth=0)
    
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    #ax.set_xlabel(r"x$_1$", fontsize=14)
    #ax.set_ylabel(r"x$_2$", fontsize=14)

    ax.tick_params(which="both", direction="inout", bottom=True, left=True, labelsize=14, pad=5, length=4, width=1)
    ax.tick_params(which="major", length=6)
    ax.set_xticks([-rmax, 0, rmax])
    ax.set_yticks([-rmax, 0, rmax])
    ax.set_aspect('equal', adjustable='box')
    ax.minorticks_on()
    return ax


##### Misc math

def polar_to_cartesian_df(df, x_key, y_key, r_key, theta_key):
    df[x_key] = df[r_key]*np.cos(df[theta_key])
    df[y_key] = df[r_key]*np.sin(df[theta_key])
    return df

def cartesian_to_polar_df(df, x_key, y_key, r_key, theta_key):
    df[r_key] = np.sqrt(df[x_key]**2 + df[y_key]**2)
    df[theta_key] = np.arctan(df[y_key]/df[x_key])
    #Get angles in range 0, 2pi
    df_copy = df.copy()
    mask1 = df_copy[x_key] < 0
    df.loc[mask1, theta_key] = df[theta_key] + np.pi
    df_copy = df.copy()
    mask2 = df_copy[theta_key] < 0
    df.loc[mask2, theta_key] = df[theta_key] + 2*np.pi
    return df

#### Metrics

def calculate_metrics(test_dfs:list, grid_dfs:list, n_data:list, truth_data, truth_test_data, pred_key, prob_key, error_key, n_max=-1):
    keys = ["N data"]
    scores = pd.DataFrame(columns=keys)
    
    if "P_blue_given_x" in truth_data.keys():
        truth_prob_key = "P_blue_given_x"
    elif "p_c1_given_r" in truth_data.keys():
        truth_prob_key = "p_c1_given_r"
    else:
        print("Not a recognized truth file format.")
        return scores
    
    scores["N data"] = n_data
    n_plots = len(n_data)
    scores["ACC"] = [accuracy_score(test_dfs[i]["class"][0:n_max], test_dfs[i][pred_key][0:n_max], normalize=True) for i in range(n_plots)]
    scores["ROCAUC"] = [roc_auc_score(test_dfs[i]["class"][0:n_max], test_dfs[i][prob_key][0:n_max]) for i in range(n_plots)]
    scores["WD test"] = [wasserstein_distance(truth_test_data[truth_prob_key][0:n_max], test_dfs[i][prob_key][0:n_max]) for i in range(len(n_data))]
    scores["WD grid"] = [wasserstein_distance(truth_data[truth_prob_key], grid_dfs[i][prob_key]) for i in range(len(n_data))]
    scores["Avg UE"] = [test_dfs[i][error_key][0:n_max].mean() for i in range(n_plots)]
    scores["Std UE"] = [test_dfs[i][error_key][0:n_max].std() for i in range(n_plots)]
    scores["Mean KL-div test"] = [kl_div(truth_test_data[truth_prob_key][0:n_max], test_dfs[i][prob_key][0:n_max]).mean() for i in range(len(n_data))]
    scores["Mean KL-div grid"] = [kl_div(truth_data[truth_prob_key], grid_dfs[i][prob_key]).mean() for i in range(len(n_data))]
    scores["LogLoss"] = [log_loss(test_dfs[i]["class"][0:n_max], test_dfs[i][prob_key][0:n_max]) for i in range(len(n_data))]

    ece = np.zeros(len(n_data))
    mce = np.zeros(len(n_data))
    rmsce = np.zeros(len(n_data))

    for i in range(len(n_data)):
        preds = torch.Tensor(test_dfs[i][prob_key])
        target = torch.Tensor(test_dfs[i]["class"])
        bce_l1 = BinaryCalibrationError(n_bins=15, norm='l1')
        ece[i] = bce_l1(preds, target).item()
        bce_l2 = BinaryCalibrationError(n_bins=15, norm='l2')
        rmsce[i] = bce_l2(preds, target).item()
        bce_max = BinaryCalibrationError(n_bins=15, norm='max')
        mce[i] = bce_max(preds, target).item()

    scores["ECE"] = ece
    scores["MCE"] = mce
    scores["RMSCE"] = rmsce
    return scores
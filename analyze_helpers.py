import numpy as np
import os
import time
import pickle
import yaml
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import pandas as pd
from shared.plotting import (
    condition_name_map,
    label_name_map,
    calculate_correlation,
)


def columnwise_mean_std(list_of_arrays):
    """
    Compute the column-wise mean and standard deviation for a list of arrays.

    Parameters:
    - list_of_arrays: List of numpy arrays.

    Returns:
    - means: Array of column-wise means.
    - std_devs: Array of column-wise standard deviations.
    """

    # Find the length of the longest array
    max_length = max(len(arr) for arr in list_of_arrays)

    # Pad the arrays to the max length
    padded_arrays = [
        np.pad(arr, (0, max_length - len(arr)), mode="constant", constant_values=np.nan)
        for arr in list_of_arrays
    ]

    # Convert the list of padded arrays to a 2D numpy array
    stacked_array = np.vstack(padded_arrays)

    # Compute the mean and standard deviation along the first axis, ignoring NaN values
    means = np.nanmean(stacked_array, axis=0)
    std_devs = np.nanstd(stacked_array, axis=0)

    return means, std_devs


def list_recent_subdirs(dir_path="", n=10):
    # list high level subdirs in path
    subdirs = [x[0][5:] for x in os.walk(dir_path) if x[0].count("/") == 1]
    # list the creation dates of each of the subdir folders
    subdirs = [x for x in subdirs if os.path.isdir(f"{dir_path}/" + x)]
    subdirs = sorted(subdirs, key=lambda x: os.path.getmtime(f"{dir_path}/" + x))
    # print the subdirs and their creation dates in (MM/DD/YYYY HH:MM:SS) format
    print("total number of subdirs:", len(subdirs), "\n")
    print(f"last {n} subdirs:")
    for subdir in subdirs[-n:]:
        date_time = os.path.getmtime(f"{dir_path}/" + subdir)
        # print in a table-like format with the date_time in the first column and the subdir in the second column
        print(
            time.strftime("%m/%d/%Y %H:%M:%S", time.localtime(date_time)), ":", subdir
        )


def process_data(experiment_name, prefix=""):
    """
    Process the data from the experiment.
    """
    # read all pickle files in the directory
    dir = f"{prefix}/{experiment_name}/"
    # look through all subdirectories for pickle files
    files = []
    for root, dirs, filenames in os.walk(dir):
        for f in filenames:
            if f.endswith(".pkl"):
                if os.stat(os.path.join(root, f)).st_size == 0:
                    continue
                files.append(os.path.join(root, f))
                with open(os.path.join(root, f[:-4] + ".yaml"), "r") as f:
                    hyperparams = yaml.load(f, Loader=yaml.FullLoader)

    num_seeds = 0
    conditions = []
    datas = {}
    for file in files:
        if os.stat(file).st_size == 0:
            continue
        else:
            sub_file = file.split("/")[2]
            sub_file = sub_file.split("_")
            condition = sub_file[3]
            seed = int(sub_file[5])
            if condition not in conditions:
                conditions.append(condition)
            if seed > num_seeds:
                num_seeds = seed
            if condition not in datas:
                datas[condition] = []
            with open(file, "rb") as file:
                data = pickle.load(file)
            datas[condition].append(data)
    num_seeds += 1

    for file in files:
        with open(file, "rb") as file:
            data = pickle.load(file)

    quantities = list(datas[conditions[0]][0].keys())

    combined_data = {}
    for quantity in quantities:
        for condition in conditions:
            if quantity not in combined_data:
                combined_data[quantity] = {}
            if condition not in combined_data[quantity]:
                combined_data[quantity][condition] = []
            for seed in range(num_seeds):
                try:
                    combined_data[quantity][condition].append(
                        datas[condition][seed][quantity][condition][0]
                    )
                except:
                    pass
            # take the mean and std over all seeds for a given condition
            mean, std = columnwise_mean_std(combined_data[quantity][condition])
            std = std / np.sqrt(num_seeds)
            combined_data[quantity][condition] = (
                mean,
                std,
                combined_data[quantity][condition],
            )

    # combined data is organized as follows:
    # combined_data[quantity][condition] = (mean, std, data)
    # where mean and std are numpy arrays of the same length

    hyperparams["experiment"]["output_dir"] = "."
    if "layer-norm" in conditions:
        conditions.remove("layer-norm")
        conditions.append("layernorm")
        for quantity in quantities:
            combined_data[quantity]["layernorm"] = combined_data[quantity].pop(
                "layer-norm"
            )
    return hyperparams, combined_data, quantities, conditions


# compute the r2 and p values for the fit
def r2_score(y, y_pred):
    """Return R^2 where x and y are array-like."""
    corr_matrix = np.corrcoef(y, y_pred)
    corr = corr_matrix[0, 1]
    return corr**2


def process_corrs(quantity, experiment_names, metrics, conditions):
    all_slopes = {}
    all_means = {}
    for experiment_name in experiment_names:
        hyperparams, combined_data, quantities, _ = process_data(experiment_name)
        exp_slopes = {}
        exp_means = {}
        for idx, metric in enumerate(metrics):
            slopes, means = calculate_correlation(
                combined_data,
                hyperparams,
                conditions=conditions,
                quantities=[quantity, metric],
            )
            exp_slopes[metric] = slopes
            exp_means[metric] = means
        all_slopes[experiment_name] = exp_slopes
        all_means[experiment_name] = exp_means

    transformed = {}
    for metric in metrics:
        transformed[metric] = {}
        for condition in conditions:
            transformed[metric][condition] = [[], []]
            for experiment_name in experiment_names:
                transformed[metric][condition][0].append(
                    all_means[experiment_name][metric][condition][0]
                )
                transformed[metric][condition][1].append(
                    all_means[experiment_name][metric][condition][1]
                )
            transformed[metric][condition][0] = np.concatenate(
                transformed[metric][condition][0]
            )  # .mean(keepdims=True)
            transformed[metric][condition][1] = np.concatenate(
                transformed[metric][condition][1]
            )  # .mean(keepdims=True)
    return transformed


def plot_corr_row(row, transformed, conditions, metrics, gs, quantity="train_r"):
    all_x = []
    all_y = []
    all_label = []
    for idx, metric in enumerate(metrics):
        ax1 = plt.subplot(gs[row, idx % 5])
        for condition in conditions:
            plt.plot(
                transformed[metric][condition][1],
                transformed[metric][condition][0],
                "o",
                label=condition_name_map[condition],
            )

        # fit a line to the data using a least-squares 1st order polynomial fit
        # first put all the data into a single vector and create an "indicator" variable
        # that will tell us which data came from which source
        x = np.hstack([transformed[metric][c][1] for c in conditions])
        y = np.hstack([transformed[metric][c][0] for c in conditions])
        c = np.hstack(
            [[i] * len(transformed[metric][c][1]) for i, c in enumerate(conditions)]
        )
        all_x.append(x)
        all_label.append(metric)
        all_y = y
        # now do the fit
        p = np.polyfit(x, y, 1)
        # and plot it

        ax1.plot(
            x,
            np.polyval(p, x),
            label="Linear Fit",
            color="black",
            linestyle="--",
            alpha=0.33,
        )

        r2 = r2_score(y, np.polyval(p, x))
        pval = stats.pearsonr(x, y)[1]
        # and display them on the plot
        # if p < 0.001, we display it as p < 0.001
        if pval < 0.05:
            fontweight = "semibold"
        else:
            fontweight = "normal"
        if pval > 0.001:
            ax1.set_title(f"$R^2 = {r2:.3f}, $p = {pval:.3f}", fontweight=fontweight)
        else:
            ax1.set_title(f"$R^2 = {r2:.3f}, $p < 0.001", fontweight=fontweight)

        ax1.set_ylabel(label_name_map[quantity])
        ax1.set_xlabel(label_name_map[metric])
        # use three x ticks, each rounded to 3 significant digits
        x_mid = (x.min() + x.max()) / 2
        ax1.set_xticks([x.min(), x_mid, x.max()])
        if metric == "weight_diff":
            x = x * 100
        ax1.set_xticklabels([f"{x.min():.3f}", f"{x_mid:.3f}", f"{x.max():.3f}"])
        # remove top and right spines
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)

    all_x = np.vstack(all_x).T
    print(all_label)

    # for idx, label in enumerate(all_label):
    #     model = sm.GLM(all_y, sm.add_constant(all_x[:, idx]))
    #     results = model.fit()
    #     print(label)
    #     print(results.summary())

    model = sm.GLM(all_y, sm.add_constant(all_x))
    results = model.fit()
    print(results.summary())

    # Save the summary directly to a LaTeX file
    with open(f"model_summary_{quantity}.tex", "w") as f:
        f.write(results.summary().as_latex())

    return ax1

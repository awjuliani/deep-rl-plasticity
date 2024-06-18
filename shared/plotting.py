import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


label_name_map = {
    "train_r": "Reward (train)",
    "test_r": "Reward (test)",
    "train_l": "Episode length (train)",
    "test_l": "Episode length (test)",
    "entropy": "Policy entropy",
    "v_estimate": "Value estimate",
    "p_loss": "Policy loss",
    "v_loss": "Value loss",
    "weight_m": "Weight magnitude",
    "weight_diff": "Weight difference",
    "grad_norm": "Gradient norm",
    "dead_units": "Dead units",
    "v_error": "Value error",
}

condition_name_map = {
    "baseline": "Warm-start",
    "reset-all": "Reset-all",
    "reset-final": "Reset-final",
    "inject": "Injection",
    "crelu": "CReLU",
    "l2-norm": "L2 norm",
    "layernorm": "LayerNorm (LN)",
    "sp-5-5": "Shrink+perturb",
    "ssp-6": "Soft shrink+perturb",
    "soft-sp-6": "Soft shrink+perturb",
    "ent-0.1": "High entropy reg",
    "l2-init": "Regen reg (L2)",
    "l2-init-4": "Regen reg (L2)",
    "w2-init": "Regen reg (W2)",
    "ssp-6-ln": "Soft shrink+perturb + LN",
    "l2-init-ln": "Regen reg (L2) + LN",
    "layer-norm": "Layer norm (LN)",
    "redo-reset": "ReDo Reset",
    "redo-reset-1": "ReDo Reset (1)",
    "redo-reset-10": "ReDo Reset",
    "redo-reset-100": "ReDo Reset (100)",
    "redo-reset-1000": "ReDo Reset (1000)",
}


def plot_result(
    sub_dict,
    title,
    hyperparams,
    base_path,
    current_epoch=None,
    overview=False,
    print_title=True,
    axis=None,
    add_legend=True,
    conditions=None,
    custom_title=None,
    normalize=True,
):
    conv_offset = 20
    algo = hyperparams["experiment"]["algo"]
    trainer_params = hyperparams[f"{algo}_trainer"]
    env_params = hyperparams["environment"]
    num_epochs = trainer_params["num_epochs"]
    shift_points = trainer_params["shift_points"]
    if current_epoch is None:
        current_epoch = num_epochs

    plt.rcParams.update({"font.size": 14})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=plt.cm.tab20.colors
    )  # Set3, tab10, tab20, Dark2
    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
    else:
        fig = None

    if overview:
        # remove shift points that are above current epoch
        shift_points = [x - 50 for x in shift_points if x < current_epoch]
        # add current epoch to shift points if training is done
        if current_epoch >= num_epochs - 1:
            shift_points.append(current_epoch - 50)

    if conditions is None:
        conditions = list(sub_dict.keys())

    label_num = 0
    sub_dict = {k: sub_dict[k] for k in conditions if k in sub_dict}
    for sess_name, (mean_value, ste_value, _) in sub_dict.items():
        if len(mean_value) < current_epoch:
            x = np.arange(len(mean_value))
            xp = np.linspace(0, len(mean_value) - 1, current_epoch)
            mean_value = np.interp(xp, x, mean_value)
            ste_value = np.interp(xp, x, ste_value)

        smooth_mean_value = smooth_data(mean_value, conv_offset)
        smooth_ste_value = smooth_data(ste_value, conv_offset)

        use_label = f"({label_num}) " + condition_name_map[sess_name]
        label_num += 1
        if overview:
            # get subset of mean and ste using shift points
            smooth_mean_value = smooth_mean_value[shift_points]
            smooth_ste_value = smooth_ste_value[shift_points]
            if normalize:
                smooth_mean_value = smooth_mean_value - smooth_mean_value[0]
            # plot
            axis.plot(smooth_mean_value, label=use_label, marker="o")
            axis.fill_between(
                np.arange(len(smooth_mean_value)),
                smooth_mean_value - smooth_ste_value,
                smooth_mean_value + smooth_ste_value,
                alpha=0.075,
            )
        else:
            axis.plot(smooth_mean_value, label=use_label)
            axis.fill_between(
                np.arange(current_epoch),
                smooth_mean_value - smooth_ste_value,
                smooth_mean_value + smooth_ste_value,
                alpha=0.075,
            )

    if not overview:
        for shift_point in trainer_params["shift_points"]:
            axis.axvline(shift_point, linestyle="--", color="black", alpha=0.5)

    task_name = env_params.get("task", "")
    if print_title:
        axis.set_title(
            f"{env_params['name']} ({task_name}, {env_params['shift_type']}), {algo}"
        )
    elif custom_title is not None:
        axis.set_title(custom_title)

    if add_legend:
        leg = axis.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower center", ncols=3
        )
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

    if overview:
        axis.set_xlabel("Round")
        axis.set_xticks(np.arange(0, len(shift_points)))
    else:
        axis.set_xlabel("Epoch (in 1000s)")
        round_length = current_epoch / (len(shift_points) + 1)
        axis.set_xticks(np.arange(0, current_epoch + round_length, round_length))
        axis.set_xticklabels(
            (np.arange(0, current_epoch + round_length, round_length) / 1000).astype(
                int
            )
        )
    axis.set_ylabel(label_name_map[title])

    # disable top and right spines
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    # save_figure(title, base_path)  # Ensure this function is compatible with the new approach
    # plt.show()
    # plt.close(fig)
    return axis


def smooth_data(data, conv_offset):
    # Smoothing using convolution and adjusting the boundary values
    smooth_data = np.convolve(data, np.ones(conv_offset) / conv_offset, mode="same")
    smooth_data[:conv_offset] = data[:conv_offset]
    smooth_data[-conv_offset:] = data[-conv_offset:]
    return smooth_data


def save_figure(title, base_path, format="png"):
    fig_folder = f"{base_path}/figures/"
    os.makedirs(fig_folder, exist_ok=True)

    plt.savefig(f"{fig_folder}/{title}.{format}", bbox_inches="tight", dpi=300)


def plot_metric(metrics, axis, condition, metric_name, x, legend=True, baseline=False):
    m_mean, m_ste = metrics
    if x % 1 == 0:
        use_label = f"({x}) " + condition_name_map[condition]
        linestyle = "-"
        marker = "o"
        alpha = 1
    else:
        use_label = None
        linestyle = ":"
        marker = "*"
        alpha = 0.65
    eb = axis.errorbar(
        x,
        m_mean,
        yerr=m_ste,
        fmt="o",
        capsize=15,
        capthick=2.5,
        elinewidth=2.5,
        label=use_label,
        linestyle=linestyle,
        alpha=alpha,
        marker=marker,
    )
    eb[-1][0].set_linestyle(linestyle)

    legend_pos = "above"
    if legend:
        if legend_pos == "above":
            axis.legend(bbox_to_anchor=(0.5, 1.2), loc="upper center", ncols=4)
        elif legend_pos == "right":
            axis.legend(bbox_to_anchor=(1.2, 0.5), loc="upper left", ncols=1)
    # axis.set_title(metric_name)
    if baseline and x == 0:
        axis.axhline(y=0, color="k", linestyle="--", alpha=0.33)


def compute_slope(metric):
    y = np.arange(len(metric))
    m, _ = np.polyfit(y, metric, 1)
    return m


def compute_slope_stats(metrics):
    ms = [compute_slope(metric) for metric in metrics]
    return np.mean(ms), np.std(ms) / np.sqrt(len(ms))


def compute_mean(metrics):
    ms = [np.mean(metric) for metric in metrics]
    return np.mean(ms), np.std(ms) / np.sqrt(len(ms))


def plot_metrics_figure(
    combined_data,
    hyperparams,
    experiment_name,
    conditions,
    quantity,
    print_title=True,
    axis=None,
    add_legend=True,
    savefig=False,
    ttest=False,
    custom_title=None,
    plot_figure=True,
):
    algo = hyperparams["experiment"]["algo"]
    trainer_params = hyperparams[f"{algo}_trainer"]
    env_params = hyperparams["environment"]
    shift_points = trainer_params["shift_points"]
    max_epoch = trainer_params["num_epochs"]
    shift_points = [x - 50 for x in shift_points]
    prefix = hyperparams["experiment"]["algo"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 14})

    if axis is None and plot_figure:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
        mean_axis = axes
    else:
        mean_axis = axis

    if plot_figure:
        task_name = env_params.get("task", "")
        if print_title:
            fig.suptitle(
                f"{env_params['name']} ({task_name}, {env_params['shift_type']}), {algo}",
                fontsize=16,
            )
        elif custom_title is not None:
            mean_axis.set_title(custom_title)

    slopes = []
    means = []
    for idx, condition in enumerate(conditions):
        _, _, quantity_data = combined_data[quantity][condition]

        items = []
        for item in quantity_data:
            if quantity == "test_r":
                x = np.arange(len(item))
                xp = np.linspace(
                    0,
                    len(item) - 1,
                    len(x) * hyperparams[f"{prefix}_trainer"]["test_interval"],
                )
                item = np.interp(xp, x, item)
            item_len = len(item)
            if item_len < max_epoch:
                continue

            # ensure that shift points are within the range of the data
            shift_points = [x for x in shift_points if x < item_len]

            # normalize metrics
            use_item = item[shift_points]
            use_item = use_item - use_item[0]
            items.append(use_item)

        slopes.append([compute_slope(item) for item in items])
        means.append([np.mean(item) for item in items])

        if plot_figure:
            plot_metric(
                compute_mean(items),
                mean_axis,
                condition,
                label_name_map[quantity],
                idx,
                add_legend,
                True,
            )

    test_cond = conditions.copy()
    test_cond = [condition_name_map[cond] for cond in test_cond]
    if plot_figure:
        mean_axis.set_ylabel(label_name_map[quantity])
        mean_axis.spines["top"].set_visible(False)
        mean_axis.spines["right"].set_visible(False)
        mean_axis.set_xticks(np.arange(len(test_cond)))
        mean_axis.set_xlabel("Condition")
        mean_axis.set_xlim(-0.5, len(test_cond) - 0.5)
        if savefig:
            os.makedirs(f"./results/{experiment_name}", exist_ok=True)
            fig.savefig(f"./results/{experiment_name}/metrics.png", bbox_inches="tight")
        return mean_axis

    if ttest:
        return run_ttests("Reset-all", test_cond, means, quantity)


def calculate_correlation(
    combined_data,
    hyperparams,
    conditions,
    quantities,
):
    algo = hyperparams["experiment"]["algo"]
    trainer_params = hyperparams[f"{algo}_trainer"]
    shift_points = trainer_params["shift_points"]
    max_epoch = trainer_params["num_epochs"]
    shift_points = [x - 50 for x in shift_points]
    prefix = hyperparams["experiment"]["algo"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 14})

    slopes = {}
    means = {}
    for idx, condition in enumerate(conditions):
        slopes[condition] = []
        means[condition] = []
        for quantity in quantities:
            _, _, quantity_data = combined_data[quantity][condition]

            items = []
            for item in quantity_data:
                if quantity == "test_r":
                    if condition == "redo-reset-10":
                        use_interval = 100
                    else:
                        use_interval = hyperparams[f"{prefix}_trainer"]["test_interval"]
                    x = np.arange(len(item))
                    xp = np.linspace(
                        0,
                        len(item) - 1,
                        len(x) * use_interval,
                    )
                    item = np.interp(xp, x, item)
                item_len = len(item)
                if item_len < max_epoch:
                    continue

                # ensure that shift points are within the range of the data
                shift_points = [x for x in shift_points if x < item_len]
                shift_points.append(max_epoch - 50)

                # normalize metrics
                use_item = item[shift_points]
                use_item = use_item - use_item[0]
                items.append(use_item)

            item_slope = [compute_slope(item) for item in items]
            item_mean = [np.mean(item) for item in items]
            item_slope = [np.mean(item_slope)]
            item_mean = [np.mean(item_mean)]
            slopes[condition].append(item_slope)
            means[condition].append(item_mean)

    return slopes, means


def run_ttests(base_condition, test_conditions, data, quantity):
    bidx = test_conditions.index(base_condition)
    print(f"T-test {quantity}")
    result_table = {}
    for idx, test_condition in enumerate(test_conditions):
        if base_condition == test_condition:
            continue
        t, p = stats.ttest_ind(data[bidx], data[idx])
        d = len(data[bidx]) + len(data[idx]) - 2
        print(f"{base_condition} vs {test_condition}: t({d}) = {t:.3}, p = {p:.3}")
        result_table[test_condition] = (t, d, p)
    return result_table

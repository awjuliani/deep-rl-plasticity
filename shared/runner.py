import numpy as np
import matplotlib.pyplot as plt
import torch
from envs.mdps import ProcGenEnv, FrameStackWrapper, GroupEnv
from envs.gridworld import GridWorld
from envs.bandits import ImageEnv
from shared.trainer import BaseTrainer
from algos.ppo.trainer import PPOTrainer
from algos.ppo.model import PPOModel
from shared.plotting import plot_result
import random
import os
import pickle
import yaml
import warnings

warnings.filterwarnings("ignore")


def make_env(env_params, seed=0, train=True):
    env_name = env_params["name"]
    shift_type = env_params["shift_type"]
    obs_type = env_params["obs_type"]
    task = env_params.get(
        "task"
    )  # Using get to avoid KeyError in case "task" is not present

    # Mapping of env_name to dataset name for ImageEnv.
    image_env_datasets = ["mnist", "fashion", "svhn", "cifar10"]

    # If the env_name matches an ImageEnv dataset
    if env_name in image_env_datasets:
        env = ImageEnv(
            shift_type=shift_type, dataset=image_env_datasets[env_name], train=train
        )
    # Other specific environment mappings
    elif env_name == "gridworld":
        env = GridWorld(
            task=task, shift_type=shift_type, train=train, seed=seed, obs_type=obs_type
        )
    elif env_name == "procgen":
        env = ProcGenEnv(shift_type=shift_type, task=task, train=train, seed=seed)
    elif env_name == "gridworld-stack":
        env = GridWorld(
            shift_type=shift_type, train=train, seed=seed, obs_type=obs_type
        )
        env = FrameStackWrapper(env, 4)
    elif env_name == "procgen-stack":
        env = ProcGenEnv(shift_type=shift_type, task=task, train=train, seed=seed)
        env = FrameStackWrapper(env, 4)
    else:
        raise ValueError("Unknown environment")
    return env


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def gen_trainer(hyperparams, seed=0, descriptor="", base_path=None):
    env_name = hyperparams["environment"]["name"]
    print(
        f"* Running session with environment: {env_name}, seed: {seed}, and: {descriptor}"
    )

    set_seeds(seed)

    algo_name = hyperparams["experiment"]["algo"]
    env_copies = hyperparams["environment"]["env_copies"]
    envs = []
    for _ in range(env_copies):
        env = make_env(hyperparams["environment"], seed=seed)
        envs.append(env)
    if env.enc_preference is not None:
        hyperparams["model"]["enc_type"] = env.enc_preference
    env = GroupEnv(envs)
    test_envs = []
    for _ in range(env_copies):
        test_env = make_env(hyperparams["environment"], seed=seed, train=False)
        test_envs.append(test_env)
    test_env = GroupEnv(test_envs)

    # grab name of last folder in base_path and set it as log_name
    log_name = base_path.split("/")[-1]

    # Define models and trainers for each algorithm
    algo_mapping = {
        "ppo": {
            "model": PPOModel,
            "trainer": PPOTrainer,
            "trainer_params_key": "ppo_trainer",
        },
    }

    # Ensure valid algorithm
    if algo_name not in algo_mapping:
        raise ValueError(f"Unknown algorithm {algo_name}")

    algo = algo_mapping[algo_name]
    model_cls = algo["model"]
    trainer_cls = algo["trainer"]
    trainer_params_key = algo["trainer_params_key"]

    model = model_cls(
        env.observation_space.shape[0],
        env.action_space.n,
        env.img_depth if hasattr(env, "img_depth") else None,
        hyperparams["model"],
    )
    trainer = trainer_cls(
        model=model,
        env=env,
        trainer_params=hyperparams[trainer_params_key],
        test_env=test_env,
        session_name=descriptor,
        experiment_name=log_name,
    )
    return trainer


def compute_stats(stat_list):
    stat_list = np.array(stat_list)
    mean = np.mean(stat_list, axis=0)
    ste = np.std(stat_list, axis=0) / np.sqrt(stat_list.shape[0])
    return mean, ste


def ensure_unique_base_path(base_path):
    """
    Ensures that the base path is unique by appending an index if it exists.
    """
    i = 1
    original_base_path = base_path
    while os.path.exists(base_path):
        base_path = f"{original_base_path}_{i}"
        i += 1
    os.makedirs(base_path)
    return base_path


def update_hyperparams(hyperparams, target, value):
    """
    Updates a nested dictionary 'hyperparams' using the '.' delimited 'target' and assigns it 'value'.
    """
    target_keys = target.split(".")
    target_dict = hyperparams
    for key in target_keys[:-1]:
        target_dict = target_dict.setdefault(key, {})
    target_dict[target_keys[-1]] = value


def run_experiments(hyperparams):
    experiment = hyperparams["experiment"]
    out_dir = experiment["output_dir"]
    experiment_name = experiment["name"]

    base_path = f"{out_dir}/results/{experiment_name}"
    if os.path.exists(base_path):
        base_path = ensure_unique_base_path(base_path)
    data_folder = f"{base_path}/data/"
    stat_list = BaseTrainer.stat_list()
    result_dict = {stat: {} for stat in BaseTrainer.stat_list()}

    for condition_name, condition_params in experiment["conditions"].items():
        for target, value in condition_params.items():
            update_hyperparams(hyperparams, target, value)

        num_sessions = experiment["num_sessions"]
        for session in range(num_sessions):
            seed = experiment["seed"] if num_sessions == 1 else session
            session_descriptor = f"{condition_name}_{session}"
            trainer = gen_trainer(
                hyperparams,
                seed=seed,
                descriptor=session_descriptor,
                base_path=base_path,
            )

            for epoch in range(trainer.num_epochs):
                stats = trainer.train(epoch)
                if (
                    epoch % experiment["save_freq"] == 0 and epoch > 0
                ) or epoch == trainer.num_epochs - 1:
                    for stat in stat_list:
                        if session == 0:
                            result_dict[stat][condition_name] = []
                        result_dict[stat][condition_name].append(stats[stat])

                    for stat in stat_list:
                        mean, ste = compute_stats(result_dict[stat][condition_name])
                        result_dict[stat][condition_name] = (mean, ste, mean)

                    save_stats(result_dict, hyperparams, data_folder)

                    for title, sub_dict in result_dict.items():
                        plot_result(sub_dict, title, hyperparams, base_path, epoch + 1)

    print(
        f"Experiment {experiment_name} is done. Results are available at {data_folder}"
    )


def save_stats(result_dict, hyperparams, data_folder):
    # save results to file
    os.makedirs(data_folder, exist_ok=True)
    with open(f"{data_folder}/{hyperparams['experiment']['name']}.pkl", "wb") as f:
        pickle.dump(result_dict, f)
    # save hyperparams to file (yaml)
    with open(f"{data_folder}/{hyperparams['experiment']['name']}.yaml", "w") as f:
        yaml.dump(hyperparams, f)
    print(f"Saved results to {data_folder}")

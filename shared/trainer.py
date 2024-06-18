import torch
from torch.utils.tensorboard import SummaryWriter
import copy


class BaseTrainer(object):
    def __init__(
        self, model, env, trainer_parameters, test_env, session_name, experiment_name
    ):
        self.model = model
        self.env = env
        self.test_env = test_env
        self.session_name = session_name
        self.num_epochs = trainer_parameters["num_epochs"]
        self.shift_points = trainer_parameters["shift_points"]
        self.test_episodes = trainer_parameters["test_episodes"]
        self.batch_size = trainer_parameters["batch_size"]
        self.test_interval = trainer_parameters["test_interval"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.init_model = copy.deepcopy(self.model)
        self.writer = SummaryWriter(
            log_dir=f"results/{experiment_name}/logs/{session_name}", max_queue=1
        )

    def shift_dist(self):
        print(f"Shifting environment distribution according to: {self.env.shift_type}")
        self.env.shift()
        self.test_env.shift()
        self.model.adapt()

    @staticmethod
    def stat_list():
        return [
            "train_r",
            "train_l",
            "v_estimate",
            "entropy",
            "test_r",
            "p_loss",
            "v_loss",
            "v_error",
            "weight_m",
            "weight_diff",
            "grad_norm",
            "dead_units",
        ]

    def collect_episode(self, env):
        # collects an episode of data from the environment
        obs = env.reset()
        done = False
        episode = []
        length = 0
        returns = 0.0
        while not done:
            action, logits, value = self.model.sample_action(obs.to(self.device))
            next_obs, reward, done, _ = env.step(action.item())
            length += 1
            returns += reward
            episode.append((obs, action, next_obs, reward, done, logits, value))
            obs = next_obs
        return episode, length, returns

    def log_stats(self, epoch, result_dict, algo):
        for stat in result_dict:
            self.writer.add_scalar(f"{algo}/{stat}", result_dict[stat][-1], epoch)
        stat_str = ", ".join(
            [f"{stat}: {result_dict[stat][-1]:.2f}" for stat in result_dict]
        )
        print(f"Epoch {epoch} " + stat_str)

    def batch_collect_episodes(self, group_envs):
        envs = group_envs.envs
        n = len(envs)

        # Initialize observations, done flags, episodes, lengths, and returns for each environment
        obs = torch.stack([env.reset() for env in envs])
        dones = torch.zeros(n, dtype=torch.bool)
        episodes = [[] for _ in range(n)]
        lengths = [0 for _ in range(n)]
        returns = [0.0 for _ in range(n)]

        while not dones.all():
            actions, logits, values = self.model.sample_action(obs.to(self.device))

            # Step each environment and collect results
            next_obses = []
            rewards = []
            new_dones = []
            for i, (o, a) in enumerate(zip(obs, actions)):
                if not dones[i]:
                    next_obs, reward, done, _ = envs[i].step(a.item())
                    next_obses.append(next_obs)
                    rewards.append(reward)
                    new_dones.append(done)
                    lengths[i] += 1
                    returns[i] += reward
                    episodes[i].append(
                        (o, a, next_obs, reward, done, logits[i], values[i])
                    )
                else:
                    next_obses.append(obs[i])
                    rewards.append(0)
                    new_dones.append(True)

            obs = torch.stack(next_obses)
            dones = torch.tensor(new_dones)

        return episodes, lengths, returns

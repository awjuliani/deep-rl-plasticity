import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from shared.trainer import BaseTrainer
from shared.modules import compute_l2_norm_difference, redo_reset, save_param_state


EPSILON = 1e-8


class PPOTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        env,
        trainer_params,
        test_env,
        session_name,
        experiment_name,
    ):
        super().__init__(
            model, env, trainer_params, test_env, session_name, experiment_name
        )
        self.gamma = trainer_params["gamma"]
        self.lamda = trainer_params["lambda"]
        self.buffer_size = trainer_params["buffer_size"]
        self.ent_coef = trainer_params["ent_coef"]
        self.num_passes = trainer_params["num_passes"]
        self.clip_param = trainer_params["clip_param"]
        self.result_dict = {stat: [] for stat in self.stat_list()}

    def train(self, epoch):
        if epoch in self.shift_points:
            self.shift_dist()
        buffer_obs = []
        buffer_actions = []
        buffer_logits = []
        buffer_values = []
        buffer_advantages = []
        epoch_train_returns = []
        buffer_lengths = []
        while len(buffer_obs) < self.buffer_size:
            batch_episodes, batch_lengths, batch_returns = self.batch_collect_episodes(
                self.env
            )

            for ep_info, ep_length, ep_return in zip(
                batch_episodes, batch_lengths, batch_returns
            ):
                (
                    ep_obs,
                    ep_actions,
                    ep_nexts,
                    ep_rewards,
                    ep_dones,
                    ep_logits,
                    ep_values,
                ) = zip(*ep_info)

                epoch_train_returns.append(ep_return)
                buffer_lengths.append(ep_length)
                buffer_logits.extend(ep_logits)
                buffer_values.extend(ep_values)
                ep_values = list(torch.stack(ep_values).detach())
                ep_rewards = self.gae(ep_rewards, ep_values, 0.0, ep_dones)
                buffer_obs.extend(ep_obs)
                buffer_actions.extend(ep_actions)
                buffer_advantages.extend(ep_rewards)
        mean_lengths = torch.tensor(buffer_lengths, dtype=torch.float32).mean()
        mean_returns = torch.tensor(epoch_train_returns, dtype=torch.float32).mean()
        self.result_dict["train_r"].append(mean_returns)
        self.result_dict["train_l"].append(mean_lengths)
        buffer_obs = torch.stack(buffer_obs)
        buffer_actions = torch.tensor(buffer_actions)
        buffer_advantages = torch.tensor(buffer_advantages, dtype=torch.float32)
        buffer_logits = torch.stack(buffer_logits)
        buffer_values = torch.stack(buffer_values)
        self.result_dict["v_estimate"].append(buffer_values.mean().item())
        model_copy = save_param_state(self.model)
        loss = self.update_model(
            buffer_obs,
            buffer_actions,
            buffer_advantages,
            buffer_logits,
            buffer_values,
            epoch,
        )
        diff_l2_norm = compute_l2_norm_difference(self.model, model_copy)
        epoch_test_returns = []
        if epoch % self.test_interval == 0:
            with torch.no_grad():
                all_br = []
                while len(all_br) < self.test_episodes:
                    _, _, batch_returns = self.batch_collect_episodes(self.test_env)
                    all_br.extend(batch_returns)
                epoch_test_returns.extend(batch_returns)
                mean_test_return = torch.tensor(
                    epoch_test_returns, dtype=torch.float32
                ).mean()
                self.result_dict["test_r"].append(mean_test_return)
        self.result_dict["p_loss"].append(loss[0])
        self.result_dict["v_loss"].append(loss[1])
        self.result_dict["weight_m"].append(self.model.weight_magnitude)
        self.result_dict["weight_diff"].append(diff_l2_norm)
        self.result_dict["entropy"].append(loss[2])
        self.result_dict["grad_norm"].append(loss[3])
        self.result_dict["v_error"].append(loss[4])
        self.result_dict["dead_units"].append(loss[5])
        self.log_stats(epoch, self.result_dict, "ppo")
        return self.result_dict

    def update_model(
        self,
        buffer_obs,
        buffer_actions,
        buffer_advantages,
        buffer_logits,
        buffer_values,
        epoch,
    ):
        buffer_advantages = buffer_advantages.to(self.device).detach()
        buffer_actions = buffer_actions.to(self.device).detach()
        value_targets = buffer_advantages + buffer_values.detach()
        old_log_probs = F.log_softmax(buffer_logits, dim=-1).detach()
        old_log_probs = old_log_probs.gather(1, buffer_actions.unsqueeze(1)).squeeze(1)

        total_pg_loss = []
        total_v_loss = []
        total_v_error = []
        total_e_loss = []
        total_grad_norm = []
        total_dead_units = []
        for _ in range(self.num_passes):
            # Shuffle the data and iterate over minibatches
            minibatch = torch.randperm(len(buffer_obs))
            for i in range(0, len(buffer_obs), self.batch_size):
                batch = minibatch[i : i + self.batch_size]
                if len(batch) < self.batch_size:
                    continue
                batch_obs = buffer_obs[batch]
                batch_actions = buffer_actions[batch]
                batch_value_target = value_targets[batch]
                batch_adv = buffer_advantages[batch]
                batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + EPSILON)
                batch_old_log_probs = old_log_probs[batch]
                batch_logits, batch_values, batch_dead = self.model(
                    batch_obs.to(self.device), check=True
                )
                batch_new_log_probs = F.log_softmax(batch_logits, dim=-1)
                batch_new_log_probs = batch_new_log_probs.gather(
                    1, batch_actions.unsqueeze(1)
                ).squeeze(1)
                batch_entropy = dist.Categorical(logits=batch_logits).entropy().mean()
                # compute the clipped policy loss
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                clip_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surr1 = ratio * batch_adv
                surr2 = clip_ratio * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # compute the value loss
                value_loss = F.mse_loss(batch_values, batch_value_target)
                with torch.no_grad():
                    value_error = (batch_value_target - batch_values).mean()

                loss = policy_loss + 0.5 * value_loss + self.ent_coef * -batch_entropy
                if self.model.l2_init > 0:
                    # fetch the params of the model and init_model
                    # calculate the L2 between the params of init_model and model
                    params = torch.cat([p.view(-1) for p in self.model.parameters()])
                    params_0 = torch.cat(
                        [p.view(-1) for p in self.init_model.parameters()]
                    )
                    l2 = torch.norm(params - params_0.detach(), 2)
                    loss += self.model.l2_init * l2
                if self.model.w2_init > 0:
                    # calculate the W2 (wasserstein distance) between the params of init_model and model
                    # first sort the params within each layer of init_model and model with ascending order
                    # then calculate the L2 between the sorted params
                    sorted_params = [
                        torch.sort(p.view(-1))[0] for p in self.model.parameters()
                    ]
                    sorted_params_0 = [
                        torch.sort(p.view(-1))[0] for p in self.init_model.parameters()
                    ]
                    sorted_params = torch.cat(sorted_params)
                    sorted_params_0 = torch.cat(sorted_params_0)
                    w2 = torch.norm(sorted_params - sorted_params_0.detach(), 2)
                    loss += self.model.w2_init * w2
                self.model.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.model.optimizer.step()
                if self.model.adapt_method == "soft-sp":
                    self.model._shrink_perturb()
                total_pg_loss.append(policy_loss.item())
                total_v_loss.append(value_loss.item())
                total_e_loss.append(batch_entropy.item())
                total_grad_norm.append(grad_norm.item())
                total_v_error.append(value_error.item())
                total_dead_units.append(batch_dead.item())
        pg_loss = np.mean(total_pg_loss)
        v_loss = np.mean(total_v_loss)
        e_loss = np.mean(total_e_loss)
        grad_norm = np.mean(total_grad_norm)
        v_error = np.mean(total_v_error)
        dead_units = np.mean(total_dead_units)
        if (
            self.model.redo_weight > 0
            and epoch % self.model.redo_freq == 0
            and epoch > 0
        ):
            redo_reset(self.model, batch_obs.to(self.device), self.model.redo_weight)
        return pg_loss, v_loss, e_loss, grad_norm, v_error, dead_units

    def gae(self, rewards, values, next_value, dones):
        # generalized advantage estimation
        values = values + [next_value]
        gae = 0
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.lamda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math


def gen_encoder(obs_size, h_size, depth, enc_type, activation="relu", layernorm=False):
    encoders = {
        "conv32": lambda: ConvEncoder(h_size, depth, 32, activation, layernorm),
        "conv64": lambda: ConvEncoder(h_size, depth, 64, activation, layernorm),
        "conv11": lambda: ConvEncoder(h_size, depth, 11, activation, layernorm),
        "linear": lambda: LinearEncoder(obs_size, h_size, activation, layernorm),
    }

    return encoders.get(enc_type, lambda: raise_not_implemented_error())()


def raise_not_implemented_error():
    raise NotImplementedError


def calculate_l2_norm(parameters):
    l2_norm_squared = 0.0
    num_params = 0
    for param in parameters:
        l2_norm_squared += torch.sum(param**2).item()
        num_params += param.numel()
    return torch.sqrt(torch.tensor(l2_norm_squared) / num_params).item()


def save_param_state(model):
    initial_state = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }
    return initial_state


def compute_l2_norm_difference(model, initial_state):
    l2_diff = 0.0
    num_params = 0

    for name, param in model.named_parameters():
        if name in initial_state:
            param_diff = param - initial_state[name]
            l2_diff += torch.sum(param_diff**2).item()
            num_params += param.numel()
        else:
            raise ValueError(
                f"Saved initial state does not contain parameters '{name}'."
            )

    return torch.sqrt(torch.tensor(l2_diff) / num_params).item()


def clone_model_state(model: nn.Module):
    return {key: value.clone().detach() for key, value in model.state_dict().items()}


def restore_model_state(model: nn.Module, state_dict: dict):
    model.load_state_dict(state_dict)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CReLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            # FF layer
            return torch.cat([F.relu(x), F.relu(-x)], dim=-1)
        else:
            # Conv layer
            return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class Injector(nn.Module):
    def __init__(self, original, in_size=256, out_size=10):
        super(Injector, self).__init__()
        if type(original) == nn.Linear:
            self.original = original
        elif type(original) == Injector:
            self.original = nn.Linear(in_size, out_size)
            aw = original.original.weight
            bw = original.new_a.weight
            cw = original.new_b.weight
            self.original.weight = nn.Parameter(aw + bw - cw)
            ab = original.original.bias
            bb = original.new_a.bias
            cb = original.new_b.bias
            self.original.bias = nn.Parameter(ab + bb - cb)
        else:
            raise NotImplementedError
        self.new_a = nn.Linear(in_size, out_size)
        self.new_b = copy.deepcopy(self.new_a)

    def forward(self, x):
        return self.original(x) + self.new_a(x) - self.new_b(x).detach()


class LnReLU(nn.Module):
    # Simple layernorm followed by RELU
    def __init__(self, h_size):
        super().__init__()
        self.norm = nn.LayerNorm(h_size)

    def forward(self, x):
        return F.relu(self.norm(x))


class GnReLU(nn.Module):
    # Simple groupnorm followed by RELU
    def __init__(self, h_size, groups=1):
        super().__init__()
        self.norm = nn.GroupNorm(groups, h_size)

    def forward(self, x):
        return F.relu(self.norm(x))


class ConvEncoder(nn.Module):
    def __init__(self, h_size, depth, conv_size, activation="relu", layernorm=False):
        super().__init__()
        self.depth = depth
        self.h_size = h_size
        self.activation = activation
        self.layernorm = layernorm
        self.encoder = getattr(self, f"conv{conv_size}")()

    def map_conv_activation(self, out_channels):
        return map_activation(self.activation, out_channels, False, self.layernorm)

    def conv11(self):
        if self.activation != "crelu":
            conv_depths = [16, 16, 32, 32, 64, self.h_size]
        else:
            conv_depths = [8, 16, 16, 32, 32, self.h_size // 2]
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 11, 11)),
            nn.Conv2d(self.depth, conv_depths[0], 3, 1, 0),
            self.map_conv_activation(conv_depths[0]),
            nn.Conv2d(conv_depths[1], conv_depths[2], 3, 1, 0),
            self.map_conv_activation(conv_depths[2]),
            nn.Conv2d(conv_depths[3], conv_depths[4], 3, 1, 0),
            self.map_conv_activation(conv_depths[4]),
            nn.Flatten(1, -1),
            nn.Linear(1600, conv_depths[5]),
            map_activation(self.activation, self.h_size, self.layernorm),
        )

    def conv32(self):
        if self.activation != "crelu":
            conv_depths = [16, 16, 32, 32, 64, self.h_size]
        else:
            conv_depths = [8, 16, 16, 32, 32, self.h_size // 2]
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 32, 32)),
            nn.Conv2d(self.depth, conv_depths[0], 4, 2, 1),
            self.map_conv_activation(conv_depths[0]),
            nn.Conv2d(conv_depths[1], conv_depths[2], 4, 2, 1),
            self.map_conv_activation(conv_depths[2]),
            nn.Conv2d(conv_depths[3], conv_depths[4], 4, 2, 1),
            self.map_conv_activation(conv_depths[4]),
            nn.Flatten(1, -1),
            nn.Linear(1024, conv_depths[5]),
            map_activation(self.activation, self.h_size, self.layernorm),
        )

    def conv64(self):
        if self.activation != "crelu":
            conv_depths = [32, 32, 64, 64, 128, 128, 128, self.h_size]
        else:
            conv_depths = [16, 32, 32, 64, 64, 128, 64, self.h_size // 2]
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 64, 64)),
            nn.Conv2d(self.depth, conv_depths[0], 4, 2, 1),
            self.map_conv_activation(conv_depths[0]),
            nn.Conv2d(conv_depths[1], conv_depths[2], 4, 2, 1),
            self.map_conv_activation(conv_depths[2]),
            nn.Conv2d(conv_depths[3], conv_depths[4], 4, 2, 1),
            self.map_conv_activation(conv_depths[4]),
            nn.Conv2d(conv_depths[5], conv_depths[6], 4, 2, 1),
            self.map_conv_activation(conv_depths[6]),
            nn.Flatten(1, -1),
            nn.Linear(2048, conv_depths[7]),
            map_activation(self.activation, self.h_size, self.layernorm),
        )

    def calc_dead_units(self, x):
        if self.activation in ["ln-relu", "relu", "crelu"]:
            return torch.mean((x == 0).float())
        elif self.activation == "tanh":
            return torch.mean((torch.abs(x) > 0.99).float())
        elif self.activation == "sigmoid":
            return torch.mean(((x < 0.01) | (x > 0.99)).float())
        else:
            raise NotImplementedError

    def forward(self, x, check=False):
        x = self.encoder(x)

        if check:
            dead_units = self.calc_dead_units(x)
            return x, dead_units
        else:
            return x


def get_activation(activation_name):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "crelu": CReLu(),
    }

    if activation_name not in activations:
        raise NotImplementedError
    return activations[activation_name]


def map_activation(activation, h_size, layernorm=False, groupnorm=False):
    layers = []

    if layernorm:
        layers.append(nn.LayerNorm(h_size))
    elif groupnorm:
        if activation == "crelu":
            h_size = h_size * 2
        layers.append(nn.GroupNorm(1, h_size))

    layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class LinearEncoder(nn.Module):
    def __init__(self, obs_size, h_size, activation="relu", layernorm=False):
        super().__init__()

        if activation == "crelu":
            h_size = h_size // 2
            h_size_mid = h_size * 2
        else:
            h_size_mid = h_size

        self.enc_a = nn.Sequential(
            nn.Linear(obs_size, h_size),
            map_activation(activation, h_size, layernorm),
        )

        self.enc_b = nn.Sequential(
            nn.Linear(h_size_mid, h_size),
            map_activation(activation, h_size, layernorm),
        )

        self.activation = activation

    def calc_dead_units(self, x):
        if self.activation in ["ln-relu", "gn-relu", "relu", "crelu"]:
            return torch.mean((x == 0).float())
        elif self.activation == "tanh":
            return torch.mean((torch.abs(x) > 0.99).float())
        elif self.activation == "sigmoid":
            return torch.mean(((x < 0.01) | (x > 0.99)).float())
        else:
            raise NotImplementedError

    def forward(self, x, check=False):
        x = self.enc_a(x)
        x = self.enc_b(x)

        if check:
            dead_units = self.calc_dead_units(x)
            return x, dead_units
        else:
            return x


def sp_module(current_module, init_module, shrink_factor, epsilon):
    use_device = next(current_module.parameters()).device
    init_params = list(init_module.to(use_device).parameters())
    for idx, current_param in enumerate(current_module.parameters()):
        current_param.data *= shrink_factor
        current_param.data += epsilon * init_params[idx].data


def mix_reset_module(current_module, init_module, mix_factor):
    # Randomly replaced units from the current module with units from the init module
    init_params = list(init_module.parameters())
    for idx, current_param in enumerate(current_module.parameters()):
        init_param = init_params[idx]
        mask = torch.rand_like(current_param.data).to(current_param.device) < mix_factor
        current_param.data = torch.where(
            mask, init_param.data.to(current_param.device), current_param.data
        )


def reinitialize_weights(module, reset_mask, next_module):
    """
    Reinitializes weights and biases of a module based on a reset mask.

    Args:
        module (torch.nn.Module): The module whose weights are to be reinitialized.
        reset_mask (torch.Tensor): A boolean tensor indicating which weights to reset.
    """
    # Reinitialize weights
    new_weights = torch.empty_like(module.weight.data)
    torch.nn.init.kaiming_uniform_(new_weights, a=math.sqrt(5))
    module.weight.data[reset_mask] = new_weights[reset_mask].to(module.weight.device)

    # Set outgoing weights to zero for reset neurons
    if type(module) == type(next_module):
        next_module.weight.data[:, reset_mask] = 0.0


def redo_reset(model, input, temp):
    """
    Apply the ReDo algorithm to selectively reset neurons in a multi-layer neural network,
    considering the model as linear/sequential for simplification.
    """
    with torch.no_grad():
        s_scores_dict = calculate_s_scores_multilayer(model, input)

        modules = [
            m
            for m in model.named_modules()
            if isinstance(m[1], (torch.nn.Linear, torch.nn.Conv2d))
        ]

        # check if there are any conv layers in the network
        has_conv = any(isinstance(m[1], torch.nn.Conv2d) for m in modules)

        for i, (name, module) in enumerate(modules):
            # Skip the first entry, which is the model itself in named_modules()
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue  # Skip non-relevant modules
            if not has_conv:
                base_name_parts = name.split(".")[:-1] + ["1.0"]
                base_name = ".".join(base_name_parts)
            elif ("policy" not in name) and ("value" not in name):
                base_name_parts = name.split(".")
                base_name_parts[-1] = str(int(base_name_parts[-1]) + 1)
                base_name_parts.append("0")
                base_name = ".".join(base_name_parts)
            else:
                continue
            if base_name in s_scores_dict:
                s_scores = s_scores_dict[base_name]
                reset_mask = s_scores <= temp

                # Check if there is a next module in the list and get it
                next_module = modules[i + 1][1] if i + 1 < len(modules) else None
                # Assuming reinitialize_weights is modified to handle the next_module
                # You would need to adjust reinitialize_weights to apply the necessary changes
                # to both the current and next modules based on reset_mask.
                reinitialize_weights(module, reset_mask, next_module)


def calculate_s_scores_multilayer(model, inputs):
    """
    Calculate the s scores for each layer of a multi-layer neural network.

    Args:
        model (torch.nn.Module): The multi-layer neural network model.
        inputs (torch.Tensor): The input distribution tensor of shape (batch_size, input_dim).

    Returns:
        dict: A dictionary where the keys are the layer names and the values are the
              corresponding s scores tensors of shape (num_neurons,).
    """
    # Create a dictionary to store the s scores for each layer
    s_scores_dict = {}

    # Register a forward hook to capture the activations of each layer
    activations = {}
    hooks = []

    def hook(module, input, output):
        activations[module] = output.detach()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            handle = module.register_forward_hook(hook)
            hooks.append(handle)

    # Forward pass through the model
    model(inputs)

    # Calculate the s scores for each layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            layer_activations = activations[module]
            s_scores = layer_activations / torch.mean(
                layer_activations, axis=1, keepdim=True
            )
            s_scores = torch.mean(s_scores, axis=0)
            if len(s_scores.shape) > 1:
                s_scores = torch.mean(
                    s_scores, axis=tuple(range(1, len(s_scores.shape)))
                )
            s_scores_dict[name] = s_scores

    # Remove the hooks to prevent memory leaks
    for handle in hooks:
        handle.remove()

    return s_scores_dict

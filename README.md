# A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning

Continual learning with deep neural networks presents challenges distinct from both the fixed-dataset and the convex continual learning regimes. One such challenge is the phenomenon of plasticity loss, wherein a neural network trained in an online fashion displays a degraded ability to fit new tasks. This problem has been extensively studied in the supervised learning and off-policy reinforcement learning (RL) settings, where a number of remedies have been proposed. In contrast, plasticity loss has received relatively less attention in the on-policy deep RL setting. Here we perform an extensive set of experiments examining plasticity loss and a variety of mitigation methods in on-policy deep RL. We demonstrate that plasticity loss also exists in this setting, and that a number of methods developed to resolve it in other settings fail, sometimes even resulting in performance that worse than performing no intervention at all. In contrast, we find that a class of "regenerative" methods are able to consistently mitigate plasticity loss in a variety of contexts. We find that in particular a continual version of shrink+perturb initialization, originally made to remedy the closely related "warm-start problem" studied in supervised learning, is able to consistently resolve plasticity loss in both gridworld tasks and more challenging environments drawn from the ProcGen and ALE RL benchmarks.

## Project overview

[`train.py`](./train.py) is the entrypoint for running individual experiments. Support for Proximal Policy Optimization (PPO) is implemented.

[`hyperparams.yaml`](./hyperparams.yaml) defines all the hyperparameters used for each experiment.

[`analyze.ipynb`](./analyze.ipynb) is a Jupyter notebook which can be used to generate the figures and tables used in the paper.

## Conditions

A Variety of intervention conditions are implemented in the codebase and can be specified in the `hyperparams.yaml` file.

* baseline: No intervention.
* reset-final: Reset the final layer(s) of the neural network.
* reset-all: Reset all layers of the neural network.
* inject: Implements the "Plasticity Injection" method to change the gradient of the final layer(s) of the neural network.
* crelu: Replaces all ReLU activations with CReLU activations.
* l2-norm: L2-normalizes the weights of the neural network.
* l2-init: Regenerative regularization with L2-norm.
* redo-reset: Implements the "ReDo" method to selectively reset the units of the neural network.
* shrink+perturb: Implements the "Shrink and Perturb" method to shrink the weights of the neural network and add noise to the weights.
* layernorm: Adds layer normalization to the neural network.

## Implemented environments

* GridWorld: NeuroNav gridworld task.
* * name - `gridworld`
* * task - `gather-vis_k` where `k` is number of unique environment configurations.
* ProcGen (name - `procgen`): Tasks from the OpenAI ProcGen environment suite.
* * name - `procgen`
* * task - `env_k` where `env` is the name of the ProcGen environment and `k` is number of unique environment configurations (`coinrun_100` for example).

## Requirements

See [`requirements.txt`](./requirements.txt).

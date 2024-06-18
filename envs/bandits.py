import gym
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10


class ImageEnv(gym.Env):
    def __init__(self, dataset="mnist", shift_type="shuffle", train=True, max_steps=10):
        if dataset == "fashion":
            data = FashionMNIST("./data", train=train, download=True)
        elif dataset == "mnist":
            data = MNIST("./data", train=train, download=True)
        elif dataset == "svhn":
            train = "train" if train else "test"
            data = SVHN("./data", split=train, download=True)
        elif dataset == "cifar10":
            data = CIFAR10("./data", train=train, download=True)
        else:
            raise ValueError("Unknown dataset")
        self.base_data = data
        self.max_steps = max_steps
        self.enc_preference = "conv64"
        if dataset in ["fashion", "mnist"]:
            self.pad = transforms.Pad(2)
            self.base_data.data = self.pad(self.base_data.data)
            self.base_data.data = torch.stack(
                [self.base_data.data, self.base_data.data, self.base_data.data], dim=1
            )
            self.targets = self.base_data.targets
        else:
            self.base_data.data = torch.tensor(self.base_data.data, dtype=torch.float32)
            if dataset == "svhn":
                self.base_data.labels = torch.tensor(
                    self.base_data.labels, dtype=torch.int64
                )
                self.targets = self.base_data.labels
            else:
                self.base_data.data = self.base_data.data.permute(0, 3, 1, 2)
                self.base_data.targets = torch.tensor(
                    self.base_data.targets, dtype=torch.int64
                )
                self.targets = self.base_data.targets
        self.shift_type = shift_type
        self.observation_space = gym.spaces.Box(0, 1, shape=(3 * 64 * 64,))
        self.action_space = gym.spaces.Discrete(10)
        self.action_map = np.arange(self.action_space.n)
        self.img_depth = 3
        if self.shift_type == "expand" and train:
            self.n = 500
            self.data = self.base_data.data[: self.n]
        elif self.shift_type == "deblur" and train:
            # shrink to 8x8 and then expand to 64x64
            self.data = transforms.Resize(8)(self.base_data.data)
            self.data = transforms.Resize(64)(self.data)
        else:
            self.data = transforms.Resize(64)(self.base_data.data)
            self.n = len(self.data)
        self.reset()

    def prepare_obs(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).reshape(-1)
        obs = obs / 255.0
        return obs

    def shift(self):
        if self.shift_type == "rotate":
            self.rotate_images()
        elif self.shift_type == "shuffle":
            self.action_map = np.random.permutation(self.action_space.n)
        elif self.shift_type == "expand":
            self.expand_set()
        elif self.shift_type == "deblur":
            self.deblur()
        elif self.shift_type == "none":
            pass
        else:
            raise ValueError("Unknown shift type")

    def deblur(self):
        self.data = self.base_data.data

    def expand_set(self):
        if self.n < len(self.base_data.data):
            self.n += 5000
        self.data = self.base_data.data[: self.n]

    def rotate_images(self):
        self.data = torch.rot90(self.data, 1, [1, 2])

    def sample_demos(self, batch_size):
        # sample a batch of images and their labels and return as x,y
        idx = np.random.choice(len(self.data), batch_size, replace=False)
        idx = np.clip(idx, 0, len(self.data) - 1)
        x = self.data[idx]
        x = self.prepare_obs(x)
        y = self.targets[idx]
        return x, y

    def reset(self):
        self.index = np.random.randint(0, len(self.data))
        self.done = False
        self.step_num = 0
        obs = self.prepare_obs(self.data[self.index])
        return obs

    def step(self, action):
        action = self.action_map[action]
        if action == self.targets[self.index]:
            reward = 1.0
        else:
            reward = 0.0
        self.step_num += 1
        if self.step_num >= self.max_steps:
            self.done = True
        else:
            # 50% chance of changing the image
            if np.random.rand() < 0.5:
                self.index = np.random.randint(0, len(self.data))
        obs = self.prepare_obs(self.data[self.index])
        return obs, reward, self.done, None

import gym
import numpy as np
import torch
from collections import deque


class ProcGenEnv(gym.Env):
    def __init__(self, shift_type="none", task="fruitbot", train=True, seed=0) -> None:
        super().__init__()
        self.seed = seed
        self.task, self.parsed_ws = self.parse_task_args(task)
        if self.parsed_ws is None:
            self.parsed_ws = 10
        self.env = self.create_env()
        self.observation_space = gym.spaces.Box(0, 1, shape=(3 * 64 * 64,))
        self.action_space = gym.spaces.Discrete(9)
        self.img_depth = 3
        self.shift_type = shift_type
        self.action_mapping = np.arange(self.action_space.n)
        self.enc_preference = "conv64"
        self.step_limit = 200
        self.grid_size = 16
        self.obs_mappings = np.arange(self.grid_size**2)
        if self.shift_type == "permute":
            self.permute_obs_mappings()
        self.train = train
        self.action_repeat = 2
        if self.train:
            self.min_level = 0
            self.max_level = self.parsed_ws
        else:
            self.min_level = 10000
            self.max_level = 10000 + self.parsed_ws
        self.difficulty = "easy"

    def create_env(self, start_level=0, num_levels=10, difficulty="easy"):
        return gym.make(
            f"procgen:procgen-{self.task}-v0",
            start_level=start_level,
            num_levels=num_levels,
            paint_vel_info=True,
            distribution_mode=difficulty,
            rand_seed=self.seed,
            use_backgrounds=True,
            restrict_themes=False,
        )

    def process_obs(self, obs):
        obs = permute_image(
            obs, obs_mapping=self.obs_mappings, grid_size=self.grid_size
        )
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        return obs.flatten() / 255.0

    def reset(self, target_level=None):
        if target_level is None:
            level_pick = np.random.randint(self.min_level, self.max_level)
        else:
            level_pick = target_level
        self.env = self.create_env(start_level=level_pick, num_levels=self.max_level)
        obs = self.env.reset()
        self.steps = 0
        return self.process_obs(obs)

    def step(self, action):
        action = self.action_mapping[action]
        reward = 0
        for _ in range(self.action_repeat):
            obs, r, done, info = self.env.step(action)
            reward += r
            if done:
                break
        self.steps += 1
        reward -= 0.01
        if self.steps > self.step_limit:
            done = True
        return (
            self.process_obs(obs),
            reward,
            done,
            None,
        )

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def permute_obs_mappings(self):
        current_seed = np.random.get_state()
        np.random.seed(self.seed)
        np.random.shuffle(self.obs_mappings)
        np.random.set_state(current_seed)

    def shift(self):
        if self.shift_type == "shuffle":
            self.action_mapping = np.random.permutation(self.action_space.n)
        elif self.shift_type == "expand":
            self.max_level += self.parsed_ws
        elif self.shift_type == "window":
            self.min_level += self.parsed_ws
            self.max_level += self.parsed_ws
        elif self.shift_type == "permute":
            self.permute_obs_mappings()
        elif self.shift_type == "difficulty":
            if self.difficulty == "easy":
                self.difficulty = "hard"
            else:
                self.difficulty = "easy"
        elif self.shift_type == "none":
            pass

    @staticmethod
    def parse_task_args(s):
        try:
            split_s = s.split("_")
            string_part = split_s[0]
            integer_part = int(split_s[1])
        except IndexError:
            return s, None
        except ValueError:
            return string_part, None
        return string_part, integer_part


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(env.observation_space.shape[0] * k,)
        )
        self.img_depth = env.img_depth * k

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return torch.cat(list(self.frames), dim=0)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class GroupEnv:
    def __init__(self, envs):
        """
        Initialize the GroupEnv with a list of environments.

        :param envs: List of environments to be managed.
        """
        self.envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.img_depth = envs[0].img_depth
        self.enc_preference = envs[0].enc_preference
        self.shift_type = envs[0].shift_type

    def shift(self):
        for env in self.envs:
            env.shift()

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        return [env.step(action) for env, action in zip(self.envs, actions)]


def permute_image(image, obs_mapping, grid_size=8):
    """
    Permute an image by breaking it into a grid and randomly shuffling the grid cells.

    Args:
    image (torch.Tensor): A 3D tensor representing the image of shape (H, W, C).
    grid_size (int): The size of the grid to break the image into (grid_size x grid_size).

    Returns:
    torch.Tensor: The permuted image.
    np.ndarray: The mapping of the original grid to the permuted grid.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # Check if the image can be evenly divided by the grid size
    h, w, _ = image.shape
    if h % grid_size != 0 or w % grid_size != 0:
        raise ValueError("Image dimensions must be evenly divisible by the grid size")

    # Calculate the height and width of each grid cell
    cell_h, cell_w = h // grid_size, w // grid_size

    # Break the image into grid cells
    cells = [
        image[i : i + cell_h, j : j + cell_w]
        for i in range(0, h, cell_h)
        for j in range(0, w, cell_w)
    ]

    # Permute the cells according to the random permutation
    permuted_cells = [cells[i] for i in obs_mapping]

    # Reconstruct the image from the permuted cells
    rows = [
        torch.cat(permuted_cells[i : i + grid_size], dim=1)
        for i in range(0, len(permuted_cells), grid_size)
    ]
    permuted_image = torch.cat(rows, dim=0)

    return permuted_image

import gym
import numpy as np
import torch
from torchvision import transforms
from neuronav.envs.grid_env import GridEnv, GridObservation, GridOrientation, GridSize
from neuronav.envs.grid_templates import GridTemplate, add_outer_blocks
import itertools


class GridWorld(gym.Env):
    # wraps a neuronav GridEnv
    def __init__(
        self, task, shift_type="none", train=True, seed=0, obs_type="linear"
    ) -> None:
        super().__init__()
        self.grid_template = GridTemplate.empty
        self.train = train
        self.seed = seed
        if obs_type == "linear" or obs_type == "conv11":
            use_obs_type = GridObservation.symbolic
        elif obs_type == "conv64":
            use_obs_type = GridObservation.visual
        self.env = GridEnv(
            obs_type=use_obs_type,
            template=self.grid_template,
            orientation_type=GridOrientation.fixed,
            size=GridSize.small,
        )

        self.enc_preference = obs_type

        if self.enc_preference == "conv64":
            self.observation_space = gym.spaces.Box(0, 1, shape=(3 * 64 * 64,))
        elif self.enc_preference == "conv11" or self.enc_preference == "linear":
            self.observation_space = gym.spaces.Box(
                0, 1, shape=(self.env.grid_size * self.env.grid_size * 3,)
            )
        self.action_space = self.env.action_space
        if self.enc_preference == "conv64":
            self.img_depth = 3
        elif self.enc_preference == "linear":
            self.img_depth = 1
        elif self.enc_preference == "conv11":
            self.img_depth = 3
        else:
            raise ValueError("Unknown encoding preference")
        self.resize = transforms.Resize(
            (64, 64), interpolation=transforms.functional.InterpolationMode.NEAREST
        )
        self.step_limit = 100

        self.shift_type = shift_type
        self.task, self.parsed_ws = self.parse_task_args(task)
        self.objects = {
            "rewards": {},
            "markers": {},
            "keys": [],
            "doors": {},
        }

        self.action_mappings = self.generate_permutations()
        self.map_idx = 0
        self.obs_mappings = np.arange(self.observation_space.shape[0])
        if self.shift_type == "permute":
            self.permute_obs_mappings()
        self.num_pits = 0
        self.num_markers = 0
        self.num_walls = 0
        self.num_gems = 0
        self.show_walls = True
        self.show_pits = True
        self.show_gems = True
        self.add_goal = True
        self.rotate = 0
        self.hue_shift = 0.0
        self.max_sides = 1
        if not self.train and self.task == "sides":
            self.max_sides = 4

        self.agent_start = (self.env.grid_size - 2, self.env.grid_size - 2)

        if self.task == "adversity":
            self.window_size = 1000
            self.num_pits = 10
            self.show_pits = False
        elif self.task == "disco":
            self.window_size = 5
            self.num_markers = 5
        elif self.task == "maze":
            if self.parsed_ws is None:
                self.window_size = 10
                self.expand_param = 10
            else:
                self.window_size = self.parsed_ws
                self.expand_param = self.parsed_ws
            self.num_walls = 15
        elif self.task == "none":
            self.window_size = 1
        elif self.task == "sides":
            self.window_size = 500
            self.agent_start = (self.env.grid_size // 2, self.env.grid_size // 2)
        elif self.task == "gather-inv" or self.task == "gather-vis":
            if self.parsed_ws is None:
                self.window_size = 10
                self.expand_param = 10
            else:
                self.window_size = self.parsed_ws
                self.expand_param = self.parsed_ws
            self.agent_start = (self.env.grid_size // 2, self.env.grid_size // 2)
            self.num_gems = 5
            self.num_pits = 5
            self.num_walls = 10
            self.add_goal = False
            if self.task == "gather-inv":
                self.show_gems = False
                self.show_pits = False
                self.show_walls = False
            else:
                self.show_pits = True
                self.show_gems = True
                self.show_walls = True
            pass
        else:
            raise ValueError("Unknown task type")
        self.set_levels()

    def generate_permutations(self):
        numbers = list(range(self.env.action_space.n))
        perms = list(itertools.permutations(numbers))
        return torch.tensor(perms, dtype=torch.int64)

    def permute_obs(self, obs):
        return obs[self.obs_mappings]

    def set_levels(self):
        if self.train:
            start = 0 + (self.seed * 10000)
        else:
            start = 1000000 + (self.seed * 10000)
        end = start + self.window_size
        self.levels = np.array([x for x in range(start, end)])

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

    def process_obs(self, obs):
        if self.enc_preference == "conv64":
            obs = obs.copy()
            obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs = transforms.functional.adjust_contrast(obs, 2.0)
            obs = transforms.functional.rotate(obs, 90 * self.rotate)
            obs = transforms.functional.adjust_hue(obs, self.hue_shift)
            obs = self.resize(obs).flatten()
            if self.shift_type == "permute":
                obs = self.permute_obs(obs)
        elif self.enc_preference == "linear" or self.enc_preference == "conv11":
            obs = obs[:, :, [0, 1, 4]]
            obs = obs.copy()
            obs = torch.tensor(obs, dtype=torch.float32).flatten()
            if self.shift_type == "permute":
                obs = self.permute_obs(obs)
        return obs

    def add_walls(self, num_walls):
        block_list = []
        block_list = add_outer_blocks(block_list, self.env.grid_size)
        for i in range(num_walls):
            added = False
            while not added:
                x = np.random.randint(2, self.env.grid_size - 2)
                y = np.random.randint(1, self.env.grid_size - 1)
                block_pos = [x, y]
                if block_pos not in block_list and block_pos != list(self.agent_start):
                    block_list.append(block_pos)
                    added = True
        self.env.blocks = block_list

    def add_markers(self, num_markers):
        self.objects["markers"] = {}
        for i in range(num_markers):
            added = False
            while not added:
                x = np.random.randint(2, self.env.grid_size - 2)
                y = np.random.randint(1, self.env.grid_size - 1)
                r, g, b = np.random.randint(32, 255, 3) / 255.0
                if (
                    (x, y) not in self.objects["markers"]
                    and [
                        x,
                        y,
                    ]
                    not in self.env.blocks
                    and (x, y) != self.agent_start
                ):
                    added = True
                    self.objects["markers"][(x, y)] = (r, g, b * 0.75)

    def add_adversities(self, num_adversities):
        for i in range(num_adversities):
            added = False
            while not added:
                x = np.random.randint(2, self.env.grid_size - 2)
                y = np.random.randint(1, self.env.grid_size - 1)
                if (
                    (x, y) not in self.objects["rewards"]
                    and [
                        x,
                        y,
                    ]
                    not in self.env.blocks
                    and (x, y) != self.agent_start
                ):
                    self.objects["rewards"][(x, y)] = [-0.75, self.show_pits, False]
                    added = True

    def add_gems(self, num_gems):
        for i in range(num_gems):
            added = False
            while not added:
                x = np.random.randint(2, self.env.grid_size - 2)
                y = np.random.randint(1, self.env.grid_size - 1)
                if (
                    (x, y) not in self.objects["rewards"]
                    and [
                        x,
                        y,
                    ]
                    not in self.env.blocks
                    and (x, y) != self.agent_start
                ):
                    self.objects["rewards"][(x, y)] = [0.75, self.show_gems, False]
                    added = True

    def place_objects(self, seed=0):
        current_seed = np.random.get_state()
        np.random.seed(seed)
        if self.add_goal:
            # first pick the side of the grid
            side = np.random.randint(self.max_sides)
            # then pick a random position on that side
            if side == 0:
                goal_pos = [1, np.random.randint(1, self.env.grid_size - 1)]
            elif side == 1:
                goal_pos = [np.random.randint(1, self.env.grid_size - 1), 1]
            elif side == 2:
                goal_pos = [
                    self.env.grid_size - 2,
                    np.random.randint(1, self.env.grid_size - 1),
                ]
            else:
                goal_pos = [
                    np.random.randint(1, self.env.grid_size - 1),
                    self.env.grid_size - 2,
                ]
            goal_pos = tuple(goal_pos)
            self.objects["rewards"] = {goal_pos: [1.0, True, True]}
        else:
            self.objects["rewards"] = {}
        self.add_walls(self.num_walls)
        self.add_adversities(self.num_pits)
        self.add_markers(self.num_markers)
        self.add_gems(self.num_gems)
        np.random.set_state(current_seed)

    def reset(self):
        self.steps = 0
        grid_level = np.random.choice(self.levels)
        self.place_objects(grid_level)
        if self.add_goal:
            tp = -0.01
        else:
            tp = 0.0
        obs = self.env.reset(
            agent_pos=self.agent_start,
            time_penalty=tp,
            objects=self.objects,
            visible_walls=self.show_walls,
        )
        return self.process_obs(obs)

    def step(self, action):
        action = self.action_mappings[self.map_idx][action]
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        if self.steps >= self.step_limit:
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
            self.map_idx += 1
            if self.map_idx >= len(self.action_mappings):
                self.map_idx = 0
        elif self.shift_type == "none":
            pass
        elif self.shift_type == "inv-vis":
            self.show_pits = not self.show_pits
        elif self.shift_type == "inv-off":
            self.num_pits = 0
        elif self.shift_type == "window":
            self.levels += self.window_size
        elif self.shift_type == "expand":
            self.window_size += self.expand_param
            self.set_levels()
        elif self.shift_type == "rotate":
            self.rotate = (self.rotate + 1) % 4
        elif self.shift_type == "permute":
            self.permute_obs_mappings()
        else:
            raise ValueError("Unknown shift type")

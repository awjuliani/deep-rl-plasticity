from envs.bandits import ImageEnv
from envs.mdps import CartPoleEnv, GridWorld


# Test shifting the ImageEnv
# ensure the environment shifts without error
# and that it still loads, resets, and steps without error
def test_image_env_shift():
    # loop over possible shift types and ensure they work
    shift_types = ["shuffle", "rotate", "expand"]
    for shift_type in shift_types:
        env = ImageEnv(shift_type=shift_type)
        env.reset()
        env.step(env.action_space.sample())
        env.shift()
        env.reset()
        env.step(env.action_space.sample())


# Test shifting the GridWorld
# ensure the environment shifts without error
# and that it still loads, resets, and steps without error
def test_grid_env_shift():
    shift_types = [
        "none",
        "shuffle",
        "adversity-visible",
        "adversity-invisible",
        "maze-invisible",
        "disco-window",
        "maze-window",
        "maze-expand",
        "maze-difficult",
    ]
    for shift_type in shift_types:
        env = GridWorld(shift_type=shift_type)
        env.reset()
        env.step(env.action_space.sample())
        env.shift()
        env.reset()
        env.step(env.action_space.sample())


# Test shifting the CartPoleEnv
# ensure the environment shifts without error
# and that it still loads, resets, and steps without error
def test_cartpole_env_shift():
    env = CartPoleEnv()
    env.reset()
    env.step(env.action_space.sample())
    env.shift()
    env.reset()
    env.step(env.action_space.sample())

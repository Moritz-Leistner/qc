import os
import pickle
import numpy as np
import gymnasium

from utils.gym_wrappers import ConvertObservations
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agxcave.agxtasks  # registers tasks

BASE = "agxcave.agxtasks.excavator"
ROCK_CONFIG = f"{BASE}.rock_capturing.config"


def calc_reward(obs):
    stone_pos = obs["stone_pos"]
    z = stone_pos[2]
    target_z = 1.7

    # distance to target height
    dist = z - target_z
    reward = -abs(dist)

    # If proper height reached
    if z >= 1.5:
        reward += 10

    return reward


def load_demo_pickles(demo_dir):
    demos = []

    for fname in sorted(os.listdir(demo_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(demo_dir, fname), "rb") as f:
            demos.append(pickle.load(f))

    return demos


def demos_to_dataset(demos):
    obs, actions, rewards, terminals, next_obs = [], [], [], [], []

    for traj in demos:
        T = len(traj)
        print(T)
        for t in range(T):
            obs.append(np.concatenate(
                [traj[t]["state"][:3], traj[t]["stone_pos"]],
                axis=-1
            ))
            actions.append(traj[t]["action"][:3])
            
            rew = calc_reward(traj[t])
            if t == T - 1:
                z_stone = traj[t]["stone_pos"][2]
                if z_stone >= 1.5:
                    rew = 200.0

            rewards.append(rew)

            terminals.append(float(t == T - 1))
            next_obs.append(np.concatenate(
                [
                    traj[t + 1]["state"][:3] if t + 1 < T else traj[t]["state"][:3],
                    traj[t + 1]["stone_pos"] if t + 1 < T else traj[t]["stone_pos"],
                ],
                axis=-1
            ))

    dataset = dict(
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.float32),
        next_observations=np.asarray(next_obs, dtype=np.float32),
        masks=1.0 - np.asarray(terminals, dtype=np.float32),
    )

    return dataset


def make_agx_env_and_dataset(env_name, demo_dir):
    # gymnasium.register(
    #     id="AgxCave-Rock-Capturing-Vision-v0",
    #     entry_point="agxcave.agxenvs:ManagerBasedEnv",
    #     disable_env_checker=True,
    #     kwargs={
    #         "env_cfg_entry_point": f"{ROCK_CONFIG}.rock_capturing_vision_cfg:RockCapturingVisionEnvCfg",
    #         "teleoperation_cfg_entry_point": f"{BASE}.teleoperation_cfg",
    #     },
    # )
    cfg = parse_env_cfg(
        env_name,
        device="cpu",
        headless=True,
        render_mode=None,
    )

    env = gymnasium.make(env_name, cfg=cfg, agx_args=[])
    env = ConvertObservations(env)
    eval_env = None

    demos = load_demo_pickles(demo_dir)
    train_dataset = demos_to_dataset(demos)

    return env, env, train_dataset, None

def convert_obs(obs):
    return np.concatenate([
        flatten_field(obs["policy"].flatten()[:3]), 
        flatten_field(obs["stone"])
    ])

def flatten_field(x):
    if x is None:
        return np.array([], dtype=np.float32)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel() 
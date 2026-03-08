import os
import pickle
import numpy as np
import gymnasium

from qc_utils.gym_wrappers import ConvertObservations, VisionObservationWrapper
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agxcave.agxtasks  # registers tasks
from rewards import calc_reward
import agxcave.agxtasks.excavator.rock_capturing.config.rock_capturing_cfg as agxrewards

BASE = "agxcave.agxtasks.excavator"
ROCK_CONFIG = f"{BASE}.rock_capturing.config"


def load_demo_pickles(demo_dir):
    demos = []

    for fname in sorted(os.listdir(demo_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(demo_dir, fname), "rb") as f:
            demos.append(pickle.load(f))

    return demos


def demos_to_dataset(demos, reward_type):
    obs, actions, rewards, terminals, next_obs = [], [], [], [], []

    for traj in demos:
        T = len(traj)
        for t in range(T):
            ob = np.concatenate(
                [traj[t]["state"][:3], traj[t]["stone_pos"]],
                axis=-1
            )
            next_ob = np.concatenate(
                [
                    traj[t + 1]["state"][:3] if t + 1 < T else traj[t]["state"][:3],
                    traj[t + 1]["stone_pos"] if t + 1 < T else traj[t]["stone_pos"],
                ],
                axis=-1
            )

            action = -25*traj[t]["action"][:3]
            reward = calc_reward(traj[t], t == T-1,reward=reward_type)

            done = (t == T - 1)
            
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            terminals.append(float(done))
            next_obs.append(next_ob)

    dataset = dict(
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.float32),
        next_observations=np.asarray(next_obs, dtype=np.float32),
    )

    dataset["masks"] = 1.0 - dataset["terminals"]

    return dataset


def make_agx_env_and_dataset(env_name, demo_dir, reward, enable_vision=False):
    cfg = parse_env_cfg(
        env_name,
        device="cpu",
        headless=True,
        render_mode=None,
    )

    reward_map = {
        1: agxrewards.RockRewards1Cfg,
        2: agxrewards.RockRewards2Cfg,
        3: agxrewards.RockRewards3Cfg,
        4: agxrewards.RockRewards4Cfg,
        5: agxrewards.RockRewards5Cfg,
    }

    if reward not in reward_map:
        raise ValueError(f"Unknown reward config: {reward}")

    cfg.rewards = reward_map[reward]()

    env = gymnasium.make(env_name, cfg=cfg, agx_args=[])
    env = VisionObservationWrapper(env) if enable_vision else ConvertObservations(env)
    # Since agx prohibits us from running two siimulations on the same thread we reuse the training env for eval
    eval_env = None

    demos = load_demo_pickles(demo_dir)
    train_dataset = demos_to_dataset(demos, reward_type=reward)

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
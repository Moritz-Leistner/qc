import os
import pickle
import numpy as np
import gymnasium
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg

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


def demos_to_dataset(demos):
    obs, actions, rewards, terminals, next_obs = [], [], [], [], []

    for traj in demos:
        T = len(traj)
        for t in range(T):
            obs.append(traj[t]["state"])
            actions.append(traj[t]["action"])
            rewards.append(0.0)
            terminals.append(float(t == T - 1))
            next_obs.append(traj[t + 1]["state"] if t + 1 < T else traj[t]["state"])

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
    gymnasium.register(
        id="AgxCave-Rock-Capturing-Vision-v0",
        entry_point="agxcave.agxenvs:ManagerBasedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{ROCK_CONFIG}.rock_capturing_vision_cfg:RockCapturingVisionEnvCfg",
            "teleoperation_cfg_entry_point": f"{BASE}.teleoperation_cfg",
        },
    )
    cfg = parse_env_cfg(
        env_name,
        device="cpu",
        headless=True,
        render_mode=None,
    )

    env = gymnasium.make(env_name, cfg=cfg)
    eval_env = None

    demos = load_demo_pickles(demo_dir)
    train_dataset = demos_to_dataset(demos)

    return env, None, train_dataset, None

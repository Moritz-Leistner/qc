import os
import pickle
import numpy as np
import gymnasium

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
        T = len(traj["actions"])
        for t in range(T):
            obs.append(traj["observations"][t])
            actions.append(traj["actions"][t])
            rewards.append(traj["rewards"][t])
            terminals.append(float(t == T - 1))
            next_obs.append(traj["observations"][t + 1] if t + 1 < T else traj["observations"][t])

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
        },
    )
    env = gymnasium.make(env_name)
    eval_env = gymnasium.make(env_name)

    demos = load_demo_pickles(demo_dir)
    train_dataset = demos_to_dataset(demos)

    return env, eval_env, train_dataset, None

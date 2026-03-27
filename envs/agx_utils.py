import os
import pickle
import random
import numpy as np
import gymnasium
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from qc_utils.gym_wrappers import ConvertObservations, VisionObservationWrapper
from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agxcave.agxtasks  # registers tasks
from rewards import calc_reward
import agxcave.agxtasks.excavator.rock_capturing.config.rock_capturing_cfg as agxrewards

BASE = "agxcave.agxtasks.excavator"
ROCK_CONFIG = f"{BASE}.rock_capturing.config"

negative_files = [
"demonstration_2026-01-18_18:11:01.pkl",
"demonstration_2026-01-18_18:11:08.pkl",
"demonstration_2026-01-18_18:23:53.pkl",
"demonstration_2026-01-18_18:40:19.pkl",
"demonstration_2026-01-18_18:11:03.pkl",
"demonstration_2026-01-18_18:21:00.pkl",
"demonstration_2026-01-18_18:39:43.pkl",
"demonstration_2026-01-18_18:47:35.pkl",
"demonstration_2026-01-18_18:40:01.pkl",
"demonstration_2026-01-18_18:48:54.pkl",
"demonstration_2026-01-18_18:11:06.pkl",
"demonstration_2026-01-18_18:44:10.pkl",
"demonstration_2026-01-18_18:49:20.pkl",
"demonstration_2026-01-18_18:30:36.pkl",
"demonstration_2026-01-18_18:31:17.pkl",
"demonstration_2026-01-18_18:23:50.pkl",
"demonstration_2026-01-18_18:31:35.pkl",
"demonstration_2026-01-18_18:52:07.pkl",
"demonstration_2026-01-18_18:10:58.pkl",
"demonstration_2026-01-18_18:44:28.pkl",
"demonstration_2026-01-18_18:23:56.pkl"
]


positive_files = ['demonstration_2026-01-18_18:11:48.pkl', 'demonstration_2026-01-18_18:12:15.pkl', 'demonstration_2026-01-18_18:12:47.pkl', 'demonstration_2026-01-18_18:13:12.pkl', 
'demonstration_2026-01-18_18:13:35.pkl', 'demonstration_2026-01-18_18:14:04.pkl', 'demonstration_2026-01-18_18:14:27.pkl', 'demonstration_2026-01-18_18:15:22.pkl', 'demonstration_2026-01-18_18:15:44.pkl',
 'demonstration_2026-01-18_18:16:06.pkl', 'demonstration_2026-01-18_18:16:31.pkl', 'demonstration_2026-01-18_18:16:55.pkl', 'demonstration_2026-01-18_18:17:13.pkl', 'demonstration_2026-01-18_18:17:41.pkl', 
 'demonstration_2026-01-18_18:18:13.pkl', 'demonstration_2026-01-18_18:18:33.pkl', 'demonstration_2026-01-18_18:19:06.pkl', 'demonstration_2026-01-18_18:19:34.pkl', 'demonstration_2026-01-18_18:20:01.pkl', 
 'demonstration_2026-01-18_18:20:20.pkl', 'demonstration_2026-01-18_18:20:48.pkl', 'demonstration_2026-01-18_18:21:21.pkl', 'demonstration_2026-01-18_18:21:44.pkl', 'demonstration_2026-01-18_18:22:05.pkl', 
 'demonstration_2026-01-18_18:22:24.pkl', 'demonstration_2026-01-18_18:22:41.pkl', 'demonstration_2026-01-18_18:22:59.pkl', 'demonstration_2026-01-18_18:23:17.pkl', 'demonstration_2026-01-18_18:23:42.pkl', 
 'demonstration_2026-01-18_18:24:48.pkl', 'demonstration_2026-01-18_18:25:10.pkl', 'demonstration_2026-01-18_18:26:07.pkl', 'demonstration_2026-01-18_18:26:43.pkl', 'demonstration_2026-01-18_18:27:04.pkl', 
 'demonstration_2026-01-18_18:27:21.pkl', 'demonstration_2026-01-18_18:27:40.pkl', 'demonstration_2026-01-18_18:27:56.pkl', 'demonstration_2026-01-18_18:28:13.pkl', 'demonstration_2026-01-18_18:28:58.pkl', 
 'demonstration_2026-01-18_18:29:34.pkl', 'demonstration_2026-01-18_18:30:10.pkl', 'demonstration_2026-01-18_18:31:03.pkl', 'demonstration_2026-01-18_18:31:49.pkl', 'demonstration_2026-01-18_18:32:12.pkl', 
 'demonstration_2026-01-18_18:32:38.pkl', 'demonstration_2026-01-18_18:33:02.pkl', 'demonstration_2026-01-18_18:33:23.pkl', 'demonstration_2026-01-18_18:33:42.pkl', 'demonstration_2026-01-18_18:34:16.pkl', 
 'demonstration_2026-01-18_18:34:51.pkl', 'demonstration_2026-01-18_18:35:12.pkl', 'demonstration_2026-01-18_18:35:41.pkl', 'demonstration_2026-01-18_18:36:00.pkl', 'demonstration_2026-01-18_18:36:30.pkl', 
 'demonstration_2026-01-18_18:36:52.pkl', 'demonstration_2026-01-18_18:37:24.pkl', 'demonstration_2026-01-18_18:37:46.pkl', 'demonstration_2026-01-18_18:38:09.pkl', 'demonstration_2026-01-18_18:38:30.pkl', 
 'demonstration_2026-01-18_18:38:51.pkl', 'demonstration_2026-01-18_18:39:15.pkl', 'demonstration_2026-01-18_18:39:31.pkl', 'demonstration_2026-01-18_18:40:41.pkl', 'demonstration_2026-01-18_18:41:09.pkl', 
 'demonstration_2026-01-18_18:41:45.pkl', 'demonstration_2026-01-18_18:42:05.pkl', 'demonstration_2026-01-18_18:42:26.pkl', 'demonstration_2026-01-18_18:43:17.pkl', 'demonstration_2026-01-18_18:43:36.pkl', 
 'demonstration_2026-01-18_18:43:51.pkl', 'demonstration_2026-01-18_18:44:53.pkl', 'demonstration_2026-01-18_18:45:08.pkl', 'demonstration_2026-01-18_18:45:26.pkl', 'demonstration_2026-01-18_18:45:45.pkl', 
 'demonstration_2026-01-18_18:46:01.pkl', 'demonstration_2026-01-18_18:46:22.pkl', 'demonstration_2026-01-18_18:46:40.pkl', 'demonstration_2026-01-18_18:46:55.pkl', 'demonstration_2026-01-18_18:47:09.pkl', 
 'demonstration_2026-01-18_18:49:34.pkl', 'demonstration_2026-01-18_18:49:50.pkl', 'demonstration_2026-01-18_18:50:12.pkl', 'demonstration_2026-01-18_18:50:28.pkl', 'demonstration_2026-01-18_18:50:47.pkl', 
 'demonstration_2026-01-18_18:51:19.pkl', 'demonstration_2026-01-18_18:51:37.pkl', 'demonstration_2026-01-18_18:52:24.pkl', 'demonstration_2026-01-18_18:52:49.pkl', 'demonstration_2026-01-18_18:53:17.pkl']


def load_demo_pickles(demo_dir, n_samples, seed=None):

    if seed is not None:
        random.seed(seed)

    pos_files = positive_files
    neg_files = negative_files

    total = len(pos_files) + len(neg_files)
    pos_ratio = len(pos_files) / total

    n_pos = round(n_samples * pos_ratio)
    n_neg = n_samples - n_pos

    n_pos = min(n_pos, len(pos_files))
    n_neg = min(n_neg, len(neg_files))

    sampled_pos = random.sample(pos_files, n_pos)
    sampled_neg = random.sample(neg_files, n_neg)

    sampled_files = sampled_pos + sampled_neg
    random.shuffle(sampled_files)

    demos = []

    for fname in sampled_files:
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(demo_dir, fname), "rb") as f:
            demos.append(pickle.load(f))

    return demos


def demos_to_dataset(demos, reward_type, wrapper, enable_vision):

    obs, actions, rewards, terminals, next_obs = [], [], [], [], []

    total_steps = sum(len(t) for t in demos)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Converting demos"),
        BarColumn(bar_width=35),
        MofNCompleteColumn(),
        TextColumn("[dim]traj {task.description}"),
        console=Console(record=True),
    ) as progress:
        task = progress.add_task("", total=len(demos))

        for i, traj in enumerate(demos):
            progress.update(task, description=f"{i+1}/{len(demos)}")
            T = len(traj)
            for t in range(T):
                ob = wrapper.observation(demo_frame_to_obs(traj[t], enable_vision))

                next_frame = traj[t + 1] if t + 1 < T else traj[t]
                next_ob = wrapper.observation(demo_frame_to_obs(next_frame, enable_vision))

                action = -25*traj[t]["action"][:3]
                reward = calc_reward(traj[t], t == T-1,reward=reward_type)

                done = (t == T - 1)
                
                obs.append(ob)
                actions.append(action)
                rewards.append(reward)
                terminals.append(float(done))
                next_obs.append(next_ob)
            
            progress.advance(task)

    dataset = dict(
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.float32),
        next_observations=np.asarray(next_obs, dtype=np.float32),
    )

    dataset["masks"] = 1.0 - dataset["terminals"]

    return dataset

def demo_frame_to_obs(frame, enable_vision=False):
    return {
        "rgb_cabine": frame["rgb_cabine"],
        "policy": frame["state"],
        "stone": frame["stone_pos"],
    } if enable_vision else {
        "policy": frame["state"],
        "stone": frame["stone_pos"],
    }

def make_agx_env_and_dataset(env_name, demo_dir, reward, enable_vision=False, max_demos=None, seed=None):
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

    demos = load_demo_pickles(demo_dir, max_demos, seed)
    train_dataset = demos_to_dataset(demos, reward, env, enable_vision)

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
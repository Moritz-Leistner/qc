from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from functools import partial
from envs.agx_utils import convert_obs


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    # fixed termination keys (same as original eval)
    termination_keys = {
        "max_steps": 0,
        "too_deep_termination": 0,
        "stone_x_distance_termination": 0,
        "stone_height_termination": 0,
        "cabin_pitch_termination": 0,
    }

    # fixed reward keys (same as original eval)
    reward_keys = {
        "rock_stable": [],
        "rock_z_pos": [],
        "rock_1_5": [],
        "rock_bucket_dis": [],
        "rock_z_pos_clipped": [],
        "energy_reg": [],
    }

    returns = []
    successes = []
    rock_lifted = []
    fall_downs = []
    end_positions = []
    episode_lengths = []

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        
        observation_history = []
        action_history = []

        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)
        
        action_queue = []

        # Custon Logging
        ep_return = 0.0
        episode_success = False
        episode_rock_lifted = False
        episode_fall_down = False
        end_position = 0.0


        gripper_contact_lengths = []
        gripper_contact_length = 0
        while not done:
            action = actor_fn(observations=observation)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step([action[0]*2,action[1]*2,action[2]*2,0,0])

            # Custom Logging
            stone_pos = next_observation[-3:]
            z = stone_pos[2]
            end_position = z

            if z >= 1.5:
                episode_rock_lifted = True

            if episode_rock_lifted and z <= 1.0:
                episode_fall_down = True

            rock_stable = info.get('extras', None)["Step_Reward/rock_stable"]
            if rock_stable > 0:
                episode_success = True

            # reward logging
            log_info = info.get("extras", {}).get("log", {})
            step_rewards = info.get("extras", {})

            for key in reward_keys.keys():
                step_key = f"Step_Reward/{key}"
                if step_key in step_rewards:
                    reward_keys[key].append(step_rewards[step_key])

            # termination logging
            if done:
                for key in termination_keys.keys():
                    term_key = f"Episode_Termination/{key}"
                    if log_info.get(term_key, 0) == 1:
                        termination_keys[key] += 1

            ep_return += reward
            # End Custom Logging

            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            # print(info)
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                # print(info["gripper_contact"])
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0


        returns.append(ep_return)        
        successes.append(episode_success)
        rock_lifted.append(episode_rock_lifted)
        fall_downs.append(episode_fall_down)
        end_positions.append(end_position)
        episode_lengths.append(step)


        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))



    for k, v in stats.items():
        arr = np.array(v)
        if np.issubdtype(arr.dtype, np.number):
            stats[k] = np.mean(arr)
        else:
            stats[k] = v
    
    # Custom Logging
    mean_return = np.mean(returns)
    success_ratio = np.mean(successes)
    over_boundary_ratio = np.mean(rock_lifted)
    fall_down_ratio = np.mean(fall_downs)
    ends = np.mean(end_positions)
    episode_length = np.mean(episode_lengths)

    stats["episode_reward"] = mean_return
    stats["episode_length"] = episode_length
    stats["success_ratio"] = success_ratio
    stats["rock_lifted_ratio"] = over_boundary_ratio
    stats["fall_down_ratio"] = fall_down_ratio
    stats["mean_end_position"] = ends

    return stats, trajs, renders


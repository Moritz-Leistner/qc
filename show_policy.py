import os
import pickle
import flax.serialization
from agents import agents

from agents.acfql import get_config as acfql_get_config
from agents.acrlpd import get_config as acrlpd_get_config
import numpy as np

from envs.agx_utils import convert_obs

from agxcave.agxenvs.utils.parse_cfg import parse_env_cfg
import agxcave.agxtasks # just to execute init for registration
import gymnasium
import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(np.random.randint(0, 2**32))

from functools import partial
from evaluation import supply_rng, evaluate


def load_agent(agent, save_dir, epoch):
    load_path = os.path.join(save_dir, f'params_{epoch}.pkl')

    with open(load_path, 'rb') as f:
        save_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(
        agent,
        save_dict['agent']
    )

    print(f'Loaded agent from {load_path}')
    return agent


# Examples could also be loaded from the demonstrations here it is hard coded how a demonstration looks like
def create_agent(agent_name, horizon_length=5, seed=42, ex_obs=np.array([-0.01555474,  0.0643234 , -0.503532  , -0.17403276,  4.9557934 , -0.08068865], dtype=np.float32), ex_action=np.array([ 1.1108594e-02,  3.7221648e-03, -6.3425303e-04, -4.4959265e-06, -5.7178527e-06], dtype=np.float32),):
    if agent_name == "acrlpd":
        config = acrlpd_get_config()
    else:
        config = acfql_get_config()

    config["horizon_length"] = horizon_length

    agent_class = agents[agent_name]
    agent = agent_class.create(
        seed,
        ex_obs,
        ex_action,
        config,
    )
    return agent


def create_env(env_name="AgxCave-Rock-Capturing-Vision-v0"):
    cfg = parse_env_cfg(
        env_name,
        device="cpu",
        headless=False,
        render_mode="human"
    )

    env = gymnasium.make(env_name, cfg=cfg, agx_args=[])
    return env


def eval_agent(agent_name, save_dir=None, epoch=None, env_name="AgxCave-Rock-Capturing-Vision-v0", seed=42, num_episodes=5, horizon_length=1):

    ex_obs = np.array([-0.01555474,  0.0643234 , -0.503532  , -0.17403276,  4.9557934 , -0.08068865], dtype=np.float32)
    ex_action = np.array([ 1.1108594e-02,  3.7221648e-03, -6.3425303e-04], dtype=np.float32)

    agent = create_agent(agent_name, horizon_length, seed=seed, ex_obs=ex_obs, ex_action=ex_action)
    
    agent = load_agent(agent, save_dir, epoch)

    env = create_env(env_name)

    # evaluate(agent, env)

    # Own Implementation, because of error, i don't want to fix due to unknown sideeffects...
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    total_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        action_queue = []

        while not done:
            obs_agent = convert_obs(obs)
            # if queue is empty create new action chunk
            if len(action_queue) == 0:
                obs_input = jnp.array(np.expand_dims(obs_agent, axis=0), dtype=jnp.float32)
                action = actor_fn(obs_input)  # <- PRNGKey wird automatisch über supply_rng übergeben

                # create Chunk 
                action_chunk = np.array(action).reshape(horizon_length, -1)
                for a in action_chunk:
                    action_queue.append(a)
            
            # execute actions
            action = action_queue.pop(0)
            obs, reward, terminated, truncated, info = env.step([action[0], action[1], action[2], 0, 0])

            done = terminated or truncated
            ep_reward += reward

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward = {ep_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    eval_agent(agent_name="acrlpd", save_dir="exp/rlpd/Debug/AgxCave-Rock-Capturing-Vision-v0/sd00020260114_013240", epoch=800000, num_episodes=10, horizon_length=1)

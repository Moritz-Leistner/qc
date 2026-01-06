import os
import time
import pickle
import jax
import gymnasium
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from envs.agx_utils import make_agx_env_and_dataset
from agents import agents
from evaluation import evaluate
from log_utils import setup_wandb
from utils.datasets import Dataset
import flax.serialization
import json

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name")
flags.DEFINE_string("agent_dir", None, "Directory with saved params_*.pkl")
flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes")
flags.DEFINE_integer("poll_interval", 30, "Seconds between checkpoint checks")
flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

config_flags.DEFINE_config_file("agent", None, lock_config=True)
 
def load_agent_from_checkpoint(agent, path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return flax.serialization.from_state_dict(agent, data["agent"])

def main(_):
    assert FLAGS.env_name is not None
    assert FLAGS.agent_dir is not None
    flags_path = os.path.join(FLAGS.agent_dir, "flags.json")
    with open(flags_path, "r") as f:
        train_flags = json.load(f)

    # Create AGX env in THIS PROCESS
    env, _, train_dataset, _ = make_agx_env_and_dataset(
        FLAGS.env_name,
        demo_dir="/home/praktikum_ws2526/Documents/qc/data/demonstrations_no_images/",
    )

    run = setup_wandb(
        project="qc",
        group="eval",
        name=f"eval-{os.path.basename(FLAGS.agent_dir)}",
    )

    # Agent init (structure only, params loaded later)
    agent_cfg = FLAGS.agent
    agent_cfg["horizon_length"] = train_flags["horizon_length"]
    agent_class = agents[agent_cfg["agent_name"]]

    # Reset env once to get observation shape
    _, _ = env.reset()

    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[agent_cfg['agent_name']]
    agent = agent_class.create(
        0,
        example_batch['observations'],
        example_batch['actions'],
        agent_cfg,
    )


    seen = set()

    print("Eval worker started, waiting for checkpoints...", flush=True)

    while True:
        ckpts = sorted(
            f for f in os.listdir(FLAGS.agent_dir)
            if f.startswith("params_") and f.endswith(".pkl")
        )

        for ckpt in ckpts:
            if ckpt in seen:
                continue

            step = int(ckpt.replace("params_", "").replace(".pkl", ""))
            path = os.path.join(FLAGS.agent_dir, ckpt)

            agent = load_agent_from_checkpoint(agent, path)

            eval_info, _, _ = evaluate(
                agent=agent,
                env=env,
                action_dim=env.action_space.shape[0],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=0,
            )

            run.log(
                {f"eval/{k}": v for k, v in eval_info.items()},
                step=step,
            )

            print(f"Evaluated checkpoint {ckpt}", flush=True)
            seen.add(ckpt)

        time.sleep(FLAGS.poll_interval)

if __name__ == "__main__":
    app.run(main)

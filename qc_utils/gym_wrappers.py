import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

def space_stack(space: gym.Space, repeat: int):
    """
    Creates new Gym space that represents the original observation/action space
    repeated `repeat` times.
    """

    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise ValueError(f"Space {space} is not supported by Octo Gym wrappers.")


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Performs temporal ensembling from https://arxiv.org/abs/2304.13705
    At every timestep we execute an exponential weighted average of the last
    `pred_horizon` predictions for that timestep.
    """

    def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):
        super().__init__(env)
        self.pred_horizon = pred_horizon
        self.exp_weight = exp_weight

        self.act_history = deque(maxlen=self.pred_horizon)

        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions):
        assert len(actions) >= self.pred_horizon

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)

    def reset(self, **kwargs):
        self.act_history = deque(maxlen=self.pred_horizon)
        return self.env.reset(**kwargs)



class ConvertObservations(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf) 

    def observation(self, obs):
        return np.concatenate([
        flatten_field(obs["policy"].flatten()[:3]), 
        flatten_field(obs["stone"])
    ])

class VisionObservationWrapper(ObservationWrapper):
    def __init__(self, env, device="cpu"):
        super().__init__(env)
        self.device = device

        self.cnn = make_resnet_feature_extractor(device)

        # policy: 3 values
        # stone: 3 values
        # image features: 512
        low_dim_size = 3 + env.observation_space["stone"].shape[0]
        image_feat_size = 512

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(low_dim_size + image_feat_size,),
            dtype=np.float32
        )

    def observation(self, obs):
        low_dim = np.concatenate([
            obs["policy"].flatten()[:3],
            obs["stone"].flatten(),
        ])

        img = obs["image"]
        if img.shape[0] != 3:
            img = np.transpose(img, (2, 0, 1))

        img = resnet_preprocess(img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.cnn(img).cpu().numpy().squeeze(0)

        return np.concatenate([low_dim, img_feat]).astype(np.float32)

def flatten_field(x):
    if x is None:
        return np.array([], dtype=np.float32)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel()

def make_resnet_feature_extractor(device="cpu"):
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.to(device)
    return resnet

resnet_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
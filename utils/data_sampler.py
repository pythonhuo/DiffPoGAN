"""
Based on Diffusion-Policies-for-Offline-RL.
https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL.git
"""
import time
import math
import torch
import numpy as np


class Data_Sampler(object):
    def __init__(self, data, device, reward_tune="no"):
        self.state = torch.from_numpy(data["observations"]).float()
        self.action = torch.from_numpy(data["actions"]).float()
        self.next_state = torch.from_numpy(data["next_observations"]).float()
        reward = torch.from_numpy(data["rewards"]).view(-1, 1).float()
        self.not_done = 1.0 - torch.from_numpy(data["terminals"]).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (reward - 0.25) * 2.0
        self.reward = reward

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device),
        )

    def random_batch(self, batch_size, min_pct=0, max_pct=1, include_logprobs=False, return_indices=False):
        indices = np.random.randint(
            int(min_pct * self._size),
            int(max_pct * self._size),
            batch_size,
        )
        batch = dict(
            observations=self.state[indices],
            actions=self.action[indices],
            rewards=self.rewards[indices],
            terminals=self.not_done[indices],
            next_observations=self.next_state[indices],
        )
        # if self._store_log_probs and include_logprobs:
        #     batch['logprobs'] = self._logprobs[indices]

        if return_indices:
            return batch, indices
        else:
            return batch


def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(
        torch.tensor(trajs_rt)
    )
    reward /= rt_max - rt_min
    reward *= 1000.0
    return reward

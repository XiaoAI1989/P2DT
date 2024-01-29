import os
import pickle

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def discounted_cumsum(x, gamma):
    y = np.zeros_like(x)
    y[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        y[t] = y[t + 1] * gamma + x[t]

    return y


class SequenceBuffer:
    def __init__(self, env_name, dataset, context_len, root_dir, gamma, sample_type='uniform', pos_encoding='absolute',
                 seed=0) -> None:
        dataset_path = os.path.join(root_dir, 'dataset', f'{env_name.lower()}-{dataset}.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        self.num_trajs = len(trajectories)
        self.dataset = dataset

        self.state_dim = trajectories[0]['observations'].shape[1]
        self.action_dim = trajectories[0]['actions'].shape[1]
        self.context_len = context_len
        self.size = sum([len(traj['observations']) for traj in trajectories]) + 1  # plus one for padding zeros

        self.states = np.zeros((self.size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.action_dim), dtype=np.float32)
        self.rewards_to_go = np.zeros((self.size,), dtype=np.float32)

        self.traj_length = np.zeros(self.num_trajs, dtype=np.int32)
        self.traj_sp = np.zeros(self.num_trajs, dtype=np.int32)  # trajectory start point
        self.traj_returns = np.zeros(self.num_trajs, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

        traj_pointer = 0
        for i, traj in enumerate(trajectories):

            len_traj = len(traj["observations"])
            self.traj_length[i] = len_traj
            start_pointer = traj_pointer
            self.traj_sp[i] = start_pointer
            self.traj_returns[i] = traj["rewards"][0]
            for j in range(len_traj):
                self.states[traj_pointer] = traj["observations"][j]
                self.actions[traj_pointer] = traj["actions"][j]
                self.rewards_to_go[traj_pointer] = traj["rewards"][j]
                traj_pointer += 1

            self.rewards_to_go[start_pointer:traj_pointer] = discounted_cumsum(
                self.rewards_to_go[start_pointer:traj_pointer], gamma)
            ############################

        # different position encoding strategies
        assert pos_encoding in ['absolute', 'relative'], 'pos_encoding should be one of [absolute, relative]'
        self.pos_encoding = pos_encoding

        # different sampling strategies
        assert sample_type in ['uniform', 'traj_return',
                               'traj_length'], 'sample_type should be one of [uniform, traj_return, traj_length]'
        self.p_sample = np.ones(self.num_trajs) / self.num_trajs if sample_type == 'uniform' else self.traj_returns / \
                                                                                                  self.traj_returns.sum() if sample_type == 'traj_return' else self.traj_length / self.traj_length.sum()

        self.state_mean, self.state_std = self.states.mean(axis=0), self.states.std(axis=0)

    def __repr__(self) -> str:
        return "SequenceBuffer"

    def sample(self, batch_size):
        selected_traj = self.rng.choice(np.arange(self.num_trajs), batch_size, replace=True, p=self.p_sample)
        selected_traj_sp = self.traj_sp[selected_traj]
        selected_offset = np.floor(
            self.rng.random(batch_size) * (self.traj_length[selected_traj] - self.context_len)).astype(np.int32).clip(
            min=0)
        selected_sp = selected_traj_sp + selected_offset
        selected_ep = selected_sp + self.traj_length[selected_traj].clip(max=self.context_len)
        selected_length = selected_ep - selected_sp

        # fill the index of those padded steps with -1, so that we can fetch the last step of the corresponding item, which is zero intentionally
        selected_index = np.array([np.concatenate([np.arange(selected_sp[i], selected_ep[i]), -
        np.ones(self.context_len - selected_length[i], dtype=np.int32)]) for i in range(batch_size)])
        masks = np.array([np.concatenate([np.ones(selected_length[i], dtype=np.bool8), np.zeros(
            self.context_len - selected_length[i], dtype=np.bool8)]) for i in range(batch_size)])

        if self.pos_encoding == 'relative':
            timesteps = np.tile(np.arange(self.context_len),
                                (batch_size, 1))  # without absolute position info of the trajectory
        else:
            timesteps = np.array([np.arange(selected_offset[i], selected_offset[i] + self.context_len)
                                  for i in
                                  range(batch_size)])  # we don't care about the timestep for those padded steps

        states = torch.as_tensor(self.states[selected_index, :]).to(dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[selected_index, :]).to(dtype=torch.float32, device=device)
        rewards_to_go = torch.as_tensor(self.rewards_to_go[selected_index, None]).to(dtype=torch.float32, device=device)
        timesteps = torch.as_tensor(timesteps).to(dtype=torch.int32, device=device)
        masks = torch.as_tensor(masks).to(dtype=torch.bool, device=device)

        return states, actions, rewards_to_go, timesteps, masks

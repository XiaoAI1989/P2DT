import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# from prompt import DualPrompt
device = torch.device("cuda:0")


class MaskedCausalAttention(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.context_len = context_len

        self.q_net = nn.Linear(hidden_dim, hidden_dim)
        self.k_net = nn.Linear(hidden_dim, hidden_dim)
        self.v_net = nn.Linear(hidden_dim, hidden_dim)

        self.proj_net = nn.Linear(hidden_dim, hidden_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((context_len, context_len))
        mask = torch.tril(ones).view(1, 1, context_len, context_len)

        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        self.q_net = self.q_net.to(device)
        self.k_net = self.k_net.to(device)
        self.v_net = self.v_net.to(device)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        self.mask = self.mask.to(device)
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))

        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        self.proj_net = self.proj_net.to(device)
        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(hidden_dim, context_len, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class expert_Block(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p, token_length=20, sum_tasks=3):
        super().__init__()
        context_len = context_len + token_length
        self.attention = MaskedCausalAttention(hidden_dim, context_len, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.task_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(token_length, hidden_dim, device=device)) for i in range(sum_tasks)])

    def forward(self, x, task_id):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        task_token = self.task_tokens[task_id]

        B = x.shape[0]
        # task_token shape: (n, hidden_dim)
        task_token = task_token.unsqueeze(0)  # shape becomes: (1, n, hidden_dim)
        task_token_expanded = task_token.repeat(B, 1, 1)  # shape becomes: (B, n, hidden_dim)
        x = torch.cat((x, task_token_expanded), dim=1)
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        x = x[:, :-task_token.shape[1]]

        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_heads, n_blocks, hidden_dim, context_len, drop_p, action_space,
                 state_mean, state_std, reward_scale, max_timestep, device, prompt_len=0, n_expert_blocks=0):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        blocks = [Block(hidden_dim, 3 * context_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.expert_blocks = None
        if n_expert_blocks != 0:
            expert_blocks = nn.ModuleList(
                [expert_Block(hidden_dim, 3 * context_len, n_heads, drop_p, prompt_len).to(device) for _ in
                 range(n_expert_blocks)])
            self.expert_blocks = expert_blocks
        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_rtg = torch.nn.Linear(1, hidden_dim)
        self.embed_state = torch.nn.Linear(state_dim, hidden_dim)

        self.embed_action = torch.nn.Linear(action_dim, hidden_dim)

        # prediction heads
        self.predict_rtg = torch.nn.Linear(hidden_dim, 1)
        self.predict_state = torch.nn.Linear(hidden_dim, state_dim)
        self.predict_action = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32)
        self.reward_scale = reward_scale
        self.max_timestep = max_timestep
        self.to(device)

    def _norm_action(self, action):

        action1 = self.action_space.low + (action + 1) * (self.action_space.high - self.action_space.low) / 2
        return action1

    def _norm_state(self, state):

        state1 = (state - self.state_mean) / self.state_std
        return state1

    def _norm_reward_to_go(self, reward_to_go):
        return reward_to_go / self.reward_scale

    def __repr__(self):
        return "DecisionTransformer"

    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        return super().to(device)

    def forward(self, states, actions, rewards_to_go, timesteps, task_id):
        states = self._norm_state(states)
        rewards_to_go = self._norm_reward_to_go(rewards_to_go)
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        rewards_to_go_embeddings = self.embed_rtg(rewards_to_go) + time_embeddings

        h = torch.stack(
            (rewards_to_go_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.blocks(h)

        if self.expert_blocks is not None:

            for expert_block in self.expert_blocks:
                h = expert_block(h, task_id)

        h = h.reshape(B, T, 3, self.hidden_dim).permute(0, 2, 1, 3)

        action_preds = self.predict_action(h[:, 1])
        state_preds = self.predict_state(h[:, 2])
        reward_to_go_preds = self.predict_rtg(h[:, 2])
        action_preds = self._norm_action(action_preds)
        return state_preds, action_preds, reward_to_go_preds

    def save(self, save_name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.state_dict(), os.path.join('models', f'{save_name}.pt'))

    def load(self, load_name):
        self.load_state_dict(torch.load(os.path.join('models', f'{load_name}.pt')))

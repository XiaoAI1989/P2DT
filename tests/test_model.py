"""End-to-end checks of the DecisionTransformer prompt mechanism (requires torch + gymnasium).

Run from the repository root:
    python tests/test_model.py
or with pytest:
    pytest tests/test_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from gymnasium.spaces import Box

from model import DecisionTransformer

STATE_DIM, ACTION_DIM, HIDDEN, CTX, PROMPT, N_TASKS = 7, 3, 32, 5, 4, 3


def make_model():
    torch.manual_seed(0)
    action_space = Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
    model = DecisionTransformer(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, n_heads=1, n_blocks=1,
        hidden_dim=HIDDEN, context_len=CTX, drop_p=0.1, action_space=action_space,
        state_mean=np.zeros(STATE_DIM, np.float32), state_std=np.ones(STATE_DIM, np.float32),
        reward_scale=1000, max_timestep=50, device="cpu",
        prompt_len=PROMPT, n_expert_blocks=2, n_tasks=N_TASKS,
    )
    model.eval()  # disable dropout for deterministic comparisons
    return model


def make_inputs(batch=2, t=CTX, seed=1):
    g = torch.Generator().manual_seed(seed)
    states = torch.randn(batch, t, STATE_DIM, generator=g)
    actions = torch.randn(batch, t, ACTION_DIM, generator=g).clamp(-1, 1)
    rtg = torch.randn(batch, t, 1, generator=g)
    timesteps = torch.arange(t).unsqueeze(0).repeat(batch, 1)
    return states, actions, rtg, timesteps


@torch.no_grad()
def test_prompt_tokens_influence_predictions():
    model = make_model()
    inputs = make_inputs()
    _, before, _ = model(*inputs, task_id=0)
    for blk in model.expert_blocks:
        blk.task_tokens[0].add_(1.0)
    _, after, _ = model(*inputs, task_id=0)
    assert not torch.equal(before, after), "perturbing the active task token must change predictions"
    print("[ok] active task token influences predictions")


@torch.no_grad()
def test_other_task_tokens_do_not_leak():
    model = make_model()
    inputs = make_inputs()
    _, before, _ = model(*inputs, task_id=0)
    for blk in model.expert_blocks:
        blk.task_tokens[1].add_(100.0)
        blk.task_tokens[2].add_(100.0)
    _, after, _ = model(*inputs, task_id=0)
    assert torch.equal(before, after), "tokens of other tasks must not affect the current task"
    print("[ok] other tasks' tokens are isolated")


def test_prompt_tokens_receive_gradient():
    model = make_model()
    states, actions, rtg, timesteps = make_inputs()
    _, action_preds, _ = model(states, actions, rtg, timesteps, task_id=1)
    loss = torch.nn.functional.mse_loss(action_preds, actions)
    loss.backward()
    for blk in model.expert_blocks:
        grad = blk.task_tokens[1].grad
        assert grad is not None and grad.abs().max() > 0, "active token must receive gradient"
        assert blk.task_tokens[0].grad is None, "inactive tokens must receive no gradient"
    print("[ok] gradients flow into the active task token only")


@torch.no_grad()
def test_trajectory_causality_preserved():
    model = make_model()
    states, actions, rtg, timesteps = make_inputs()
    _, before, _ = model(states, actions, rtg, timesteps, task_id=0)
    states2 = states.clone()
    states2[:, -1] += 50.0  # perturb only the last timestep
    _, after, _ = model(states2, actions, rtg, timesteps, task_id=0)
    assert torch.equal(before[:, :-1], after[:, :-1]), \
        "future states must not influence earlier action predictions"
    print("[ok] causal structure of the trajectory is intact")


@torch.no_grad()
def test_dynamic_task_expansion():
    model = make_model()
    assert model.n_task_tokens == N_TASKS
    model.add_task_token()
    assert model.n_task_tokens == N_TASKS + 1
    inputs = make_inputs()
    _, preds, _ = model(*inputs, task_id=N_TASKS)  # forward with the newly added task works
    assert torch.isfinite(preds).all()
    for blk in model.expert_blocks:
        assert torch.equal(blk.task_tokens[-1], blk.task_tokens[-2]), \
            "new token should warm-start from the latest task's token"
    print("[ok] add_task_token() grows the prompt pool beyond the initial task count")


@torch.no_grad()
def test_prompt_len_zero_is_valid():
    torch.manual_seed(0)
    action_space = Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
    model = DecisionTransformer(
        state_dim=STATE_DIM, action_dim=ACTION_DIM, n_heads=1, n_blocks=1,
        hidden_dim=HIDDEN, context_len=CTX, drop_p=0.0, action_space=action_space,
        state_mean=np.zeros(STATE_DIM, np.float32), state_std=np.ones(STATE_DIM, np.float32),
        reward_scale=1000, max_timestep=50, device="cpu",
        prompt_len=0, n_expert_blocks=2, n_tasks=N_TASKS,
    )
    model.eval()
    _, preds, _ = model(*make_inputs(), task_id=0)
    assert preds.shape == (2, CTX, ACTION_DIM)
    print("[ok] prompt_len=0 ablation path is valid")


if __name__ == "__main__":
    test_prompt_tokens_influence_predictions()
    test_other_task_tokens_do_not_leak()
    test_prompt_tokens_receive_gradient()
    test_trajectory_causality_preserved()
    test_dynamic_task_expansion()
    test_prompt_len_zero_is_valid()
    print("all model tests passed")

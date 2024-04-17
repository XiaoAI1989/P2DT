import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from dotmap import DotMap
from gymnasium.wrappers import RecordEpisodeStatistics
from hydra.utils import instantiate
from omegaconf import OmegaConf

import utils
from buffer import SequenceBuffer
from checkpoints import load_model_checkpoint, save_model_checkpoint
from model import DecisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TASK_SPECIFIC_KEYWORDS = ("task_tokens", "embed", "predict")


def is_shared_parameter(name: str) -> bool:
    return not any(keyword in name for keyword in TASK_SPECIFIC_KEYWORDS)


def get_parameter_groups(model: DecisionTransformer):
    shared_params, task_specific_params = [], []
    for name, param in model.named_parameters():
        if is_shared_parameter(name):
            shared_params.append(param)
        else:
            task_specific_params.append(param)
    return shared_params, task_specific_params


def snapshot_shared_parameters(model: DecisionTransformer):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if is_shared_parameter(name)
    }


def build_ewc_loss(model: DecisionTransformer, ewc_snapshots, strength: float):
    if not ewc_snapshots or strength <= 0:
        return torch.zeros((), device=device)

    current_params = dict(model.named_parameters())
    ewc_loss = torch.zeros((), device=device)

    for snapshot in ewc_snapshots:
        for name, previous_param in snapshot["params"].items():
            fisher = snapshot["fisher"][name]
            ewc_loss = ewc_loss + torch.mul(
                fisher, (current_params[name] - previous_param) ** 2
            ).sum()

    return 0.5 * strength * ewc_loss


def estimate_fisher(model: DecisionTransformer, buffer: SequenceBuffer, batch_size, task_id, n_batches):
    """Diagonal Fisher information of the shared parameters, estimated after a task finishes.

    Matches the paper's definition M_k = (d log p / d w_k)^2: gradients of the
    behavior-cloning likelihood only, with the model fixed at the task's final weights.
    """
    was_training = model.training
    model.eval()
    fisher = {
        name: torch.zeros_like(param.data)
        for name, param in model.named_parameters()
        if is_shared_parameter(name)
    }
    for _ in range(n_batches):
        states, actions, rewards_to_go, timesteps, mask = buffer.sample(batch_size)
        _, action_preds, _ = model(states, actions, rewards_to_go, timesteps, task_id)
        loss = F.mse_loss(action_preds[mask], actions[mask].detach(), reduction='mean')
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2 / n_batches
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return fisher


def load_previous_task_weights(model: DecisionTransformer, checkpoint_file):
    """Carry over the shared backbone and the prompt pool from the previous task.

    Task-specific embeddings and prediction heads are NOT carried over: each
    environment has its own state/action dimensionality and keeps its own copy.
    """
    previous_dict, _ = load_model_checkpoint(checkpoint_file, map_location=device)
    own_dict = model.state_dict()
    for key, value in previous_dict.items():
        if key not in own_dict or own_dict[key].shape != value.shape:
            continue
        if is_shared_parameter(key) or "task_tokens" in key:
            own_dict[key] = value
    model.load_state_dict(own_dict)


def progressive_prompt_init(model: DecisionTransformer, task_id):
    """Warm-start the new task's prompt from the previous task's learned prompt.

    Realizes the 'progressive' transfer between consecutive task prompts; with the
    backbone nearly frozen this only changes the starting point of the new tokens.
    """
    if model.expert_blocks is None or task_id <= 0:
        return
    with torch.no_grad():
        for expert_block in model.expert_blocks:
            if task_id < len(expert_block.task_tokens):
                expert_block.task_tokens[task_id].copy_(expert_block.task_tokens[task_id - 1])


def build_continual_eval_state(model: DecisionTransformer, task_model_dict, final_shared_model_dict):
    """Combine task-specific parameters from a task checkpoint with the final shared backbone.

    This mirrors continual evaluation after the last task finishes:
    - shared parameters come from the final model after the whole task sequence
    - task-specific parameters come from the checkpoint saved when that task finished
    """
    merged_state_dict = model.state_dict()

    for key in merged_state_dict.keys():
        if key in task_model_dict and not is_shared_parameter(key):
            merged_state_dict[key] = task_model_dict[key]

    for key in merged_state_dict.keys():
        if key in final_shared_model_dict and is_shared_parameter(key):
            merged_state_dict[key] = final_shared_model_dict[key]

    return merged_state_dict


def build_checkpoint_metadata(env_name, buffer: SequenceBuffer, rtg_target):
    return {
        "env_name": env_name,
        "state_mean": buffer.state_mean,
        "state_std": buffer.state_std,
        "rtg_target": rtg_target,
    }


def load_buffer_stats(env_name, cfg, root_dir, seed, checkpoint_metadata):
    if checkpoint_metadata:
        return checkpoint_metadata["state_mean"], checkpoint_metadata["state_std"]

    buffer = SequenceBuffer(
        env_name,
        cfg.buffer.dataset,
        cfg.buffer.context_len,
        root_dir,
        cfg.buffer.gamma,
        cfg.buffer.sample_type,
        cfg.buffer.pos_encoding,
        seed,
    )
    return buffer.state_mean, buffer.state_std


def shift_context_window(states, actions, rewards_to_go, timesteps):
    states[:, :-1] = states[:, 1:].clone()
    actions[:, :-1] = actions[:, 1:].clone()
    rewards_to_go[:, :-1] = rewards_to_go[:, 1:].clone()
    timesteps[:, :-1] = timesteps[:, 1:].clone()
    actions[:, -1].zero_()


@torch.no_grad()
def eval(env: gym.vector.Env, model: DecisionTransformer, rtg_target, env_index):
    # parallel evaluation with vectorized environment
    model.eval()

    episodes = env.num_envs
    returns = np.zeros(episodes, dtype=np.float32)
    done_flags = np.zeros(episodes, dtype=np.bool_)

    state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
    act_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
    context_len = model.context_len
    max_timestep = model.max_timestep

    state, _ = env.reset(seed=[np.random.randint(0, 10000) for _ in range(episodes)])

    states = torch.zeros((episodes, context_len, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((episodes, context_len, act_dim), dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((episodes, context_len, 1), dtype=torch.float32, device=device)
    timesteps = torch.zeros((episodes, context_len), dtype=torch.long, device=device)

    reward_to_go = np.full((episodes, 1), rtg_target, dtype=np.float32)
    timestep = 0

    while not done_flags.all():
        active_mask = ~done_flags
        if timestep >= context_len:
            shift_context_window(states, actions, rewards_to_go, timesteps)
            write_index = context_len - 1
            window_len = context_len
        else:
            write_index = timestep
            window_len = timestep + 1

        states[:, write_index] = torch.as_tensor(state, dtype=torch.float32, device=device)
        rewards_to_go[:, write_index] = torch.as_tensor(reward_to_go, dtype=torch.float32, device=device)
        timesteps[:, write_index] = timestep
        _, action_preds, _ = model(
            states[:, :window_len],
            actions[:, :window_len],
            rewards_to_go[:, :window_len],
            timesteps[:, :window_len],
            env_index,
        )

        action = action_preds[:, -1]
        actions[:, write_index] = action
        action = action.detach().cpu().numpy()
        state, reward, done, truncated, _ = env.step(action)

        reward = reward.astype(np.float32)
        returns += reward * active_mask
        reward_to_go -= reward.reshape(episodes, 1) * active_mask.reshape(episodes, 1)
        done_flags = np.logical_or(done_flags, np.logical_or(done, truncated))
        timestep += 1
        if timestep >= max_timestep:
            break

    return np.mean(returns), np.std(returns)


def train(cfg, seed, log_dict, idx, logger, barrier, cmd, start_task_index=0):
    env_name_list = cfg.train.env_name
    n_tasks = len(env_name_list)
    ewc_snapshots = []
    for env_index, env_name in enumerate(env_name_list[start_task_index:], start=start_task_index):
        eval_env = gym.vector.make(
            env_name + '-v4',
            render_mode="rgb_array",
            num_envs=cfg.train.eval_episodes,
            asynchronous=False,
            wrappers=RecordEpisodeStatistics,
        )
        utils.set_seed_everywhere(eval_env, seed)

        state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
        action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)

        buffer = instantiate(cfg.buffer, env_name=env_name, root_dir=cmd, seed=seed)
        model = instantiate(
            cfg.model,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=eval_env.envs[0].action_space,
            state_mean=buffer.state_mean,
            state_std=buffer.state_std,
            n_tasks=n_tasks,
            device=device,
        )

        rtg = cfg.rtg_target[env_index]
        cfg1 = DotMap(OmegaConf.to_container(cfg.train, resolve=True))

        if env_index != 0:
            load_previous_task_weights(model, f'models/{env_index - 1}_final_model_seed_{seed}.pt')
            if cfg.train.get("progressive_prompt_init", False):
                progressive_prompt_init(model, env_index)

        shared_params, task_params = get_parameter_groups(model)
        lr_task_token = cfg.train.lr_task_token
        lr_other = cfg.train.lr_other
        optimizer = torch.optim.Adam([
            {"params": shared_params, "lr": lr_other},
            {"params": task_params, "lr": lr_task_token},
        ])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / cfg1.warmup_steps, 1),
        )

        logger.info(f"Training seed {seed} for {cfg1.timesteps} timesteps with {model} and {buffer}")

        using_mp = barrier is not None

        if using_mp:
            local_log_dict = {key: [] for key in log_dict.keys()}
        else:
            local_log_dict = log_dict
            for key in local_log_dict.keys():
                local_log_dict[key].append([])

        best_reward = -np.inf
        utils.write_to_dict(local_log_dict, 'rtg_target', rtg, using_mp)
        model.train()
        for timestep in range(1, cfg1.timesteps + 1):
            states, actions, rewards_to_go, timesteps, mask = buffer.sample(cfg1.batch_size)
            # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
            _, action_preds, _ = model.forward(states, actions, rewards_to_go, timesteps, env_index)
            action_preds = action_preds[mask]
            action_loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')
            ewc_loss = build_ewc_loss(model, ewc_snapshots, cfg.train.ewc_item)
            total_loss = action_loss + ewc_loss
            utils.write_to_dict(local_log_dict, 'action_loss', total_loss.item(), using_mp)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            if timestep % cfg1.eval_interval == 0:
                eval_mean, eval_std = eval(eval_env, model, rtg, env_index)
                model.train()
                utils.write_to_dict(local_log_dict, 'eval_steps', timestep - 1, using_mp)
                utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp)
                d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)
                utils.write_to_dict(local_log_dict, 'd4rl_score', d4rl_score, using_mp)
                logger.info(f"Seed: {seed}, Env: {env_name}, Step: {timestep}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

                if eval_mean > best_reward:
                    best_reward = eval_mean
                    save_model_checkpoint(
                        model,
                        f'{env_index}_best_model_seed_{seed}',
                        metadata=build_checkpoint_metadata(env_name, buffer, rtg),
                    )
                    logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {timestep}')

            if timestep % cfg1.plot_interval == 0:
                utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, timestep,
                                         f'{env_name} (seed {seed})', using_mp)

        if cfg.train.ewc_item > 0:
            fisher_info = estimate_fisher(
                model, buffer, cfg1.batch_size, env_index,
                cfg.train.get("fisher_batches", 100),
            )
            ewc_snapshots.append({
                "params": snapshot_shared_parameters(model),
                "fisher": fisher_info,
            })
        save_model_checkpoint(
            model,
            f'{env_index}_final_model_seed_{seed}',
            metadata=build_checkpoint_metadata(env_name, buffer, rtg),
        )

    logger.info(f"Finish training seed {seed}; last task ({env_name}) eval mean: {eval_mean}")

    return eval_mean


def final_eval(env_list, rtg_target, logger, cfg, seed, cmd, exp_path="", num=None):
    # parallel evaluation with vectorized environment
    logger.info(f'--------------------forgetting test-------------------')

    if num is None:
        num = len(env_list) - 1

    final_shared_model_dict, _ = load_model_checkpoint(
        os.path.join(exp_path, 'models', f'{num}_final_model_seed_{seed}.pt'),
        map_location=device,
    )

    for i in range(len(env_list)):
        env_name = env_list[i]
        eval_env = gym.vector.make(
            env_name + '-v4',
            render_mode="rgb_array",
            num_envs=cfg.train.eval_episodes,
            asynchronous=False,
            wrappers=RecordEpisodeStatistics,
        )
        utils.set_seed_everywhere(eval_env, seed)

        state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
        action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
        task_model_dict, task_metadata = load_model_checkpoint(
            os.path.join(exp_path, 'models', f'{i}_final_model_seed_{seed}.pt'),
            map_location=device,
        )
        state_mean, state_std = load_buffer_stats(env_name, cfg, cmd, seed, task_metadata)
        model = instantiate(
            cfg.model,
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=eval_env.envs[0].action_space,
            state_mean=state_mean,
            state_std=state_std,
            n_tasks=len(env_list),
            device=device,
        )
        merged_state_dict = build_continual_eval_state(
            model,
            task_model_dict,
            final_shared_model_dict,
        )
        model.load_state_dict(merged_state_dict, strict=False)
        rtg = task_metadata.get("rtg_target", rtg_target[i])
        eval_mean, eval_std = eval(eval_env, model, rtg, env_index=i)
        d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)

        logger.info(
            f"Seed: {seed}, env: {env_name}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}, "
            f"D4RL score: {d4rl_score:.2f}"
        )

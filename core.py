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
from model import DecisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval(env: gym.vector.Env, model: DecisionTransformer, rtg_target,env_index):
    # parallel evaluation with vectorized environment
    model.eval()

    episodes = env.num_envs
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=np.bool8)

    state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
    act_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
    max_timestep = model.max_timestep
    context_len = model.context_len
    # each vectorized environment us
    timesteps = torch.tile(torch.arange(max_timestep, device=device), (episodes, 1))

    state, _ = env.reset(seed=[np.random.randint(0, 10000) for _ in range(episodes)])

    # placeholder for states, actions, rewards_to_go
    states = torch.zeros((episodes, max_timestep, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((episodes, max_timestep, act_dim), dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep = rtg_target, 0

    while not done_flags.all():

        states[:, timestep] = torch.tensor(state)
        rewards_to_go[:, timestep] = torch.tensor(reward_to_go)
        timestep += 1
        if timestep - context_len <= 0:
            start = 0
        else:
            start = timestep - context_len
        state_preds, action_preds, reward_to_go_preds = model(states[:, start:timestep],
                                                              actions[:, start:timestep],
                                                              rewards_to_go[:, start:timestep],
                                                              timesteps[:, start:timestep],
                                                              env_index)

        action = action_preds[:, -1]
        actions[:, timestep - 1] = action
        action = action.detach().cpu().numpy()
        state, reward, done, truncated, _ = env.step(action)

        done_flags = (1-(1 - done)*(1-truncated)*(1-done_flags))
        reward=reward * (1-done_flags)
        returns += reward
        reward=reward.reshape(10,1)
        reward_to_go-=reward
        if timestep >= max_timestep:
            break
        

    return np.mean(returns), np.std(returns)


def train(cfg, seed, log_dict, idx, logger, barrier, cmd):

    env_name_list = cfg.train.env_name

    fisher_info_list=[]
    for env_index, env_name in enumerate(env_name_list):
        eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes, asynchronous=False, wrappers=RecordEpisodeStatistics)
        utils.set_seed_everywhere(eval_env, seed)

        state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
        action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)

        buffer = instantiate(cfg.buffer,env_name=env_name, root_dir=cmd, seed=seed,)
        model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim, action_space=eval_env.envs[0].action_space, state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)

        rtg=cfg.rtg_target[env_index]
        cfg1 = DotMap(OmegaConf.to_container(cfg.train, resolve=True))

        if env_index!=0:
            dic1 = model.state_dict()
            model_dict = torch.load(f'models/{env_index-1}_final_model_seed_{seed}.pt')
            for key, var in model_dict.items():
                # print(key)
                if "block" in key or "task_aware" in key:
                    dic1[key] = var
            model.load_state_dict(dic1)

            # 获取模型的所有参数，去除task_token
        other_params = [p for n, p in model.named_parameters() if "task_tokens"  not in n and "embed" not in n and "predict" not in n]

        # 获取task_token参数
        task_params = [p for n, p in model.named_parameters() if "task_tokens"  in n or "embed" in n or "predict" in n]


        # 设置大的学习率
        lr_task_token = cfg.train.lr_task_token
        # 设置小的学习率
        lr_other = cfg.train.lr_other
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": lr_other},
            {"params": task_params, "lr": lr_task_token},
        ])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda step: min((step + 1) / cfg1.warmup_steps, 1))



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
        fisher_info = {}
        for name, param in model.named_parameters():
            fisher_info[name] = torch.zeros_like(param.data)
        for timestep in range(1, cfg1.timesteps + 1):
            states, actions, rewards_to_go, timesteps, mask = buffer.sample(cfg1.batch_size)
            # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
            state_preds, action_preds, return_preds = model.forward(states, actions, rewards_to_go, timesteps,env_index)
            action_preds = action_preds[mask]
            action_loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')
            utils.write_to_dict(local_log_dict, 'action_loss', action_loss.item(), using_mp)

            if env_index!=0:

                other_params = {n: p for n, p in model.named_parameters() if
                                "task_tokens" not in n and "embed" not in n and "predict" not in n}

                ewc_loss = cfg.train.ewc_item * 0.5 * sum(
                    [torch.mul(fisher_info[p] ,(model.state_dict()[p] - other_params[p]) ** 2 ).sum() for p in other_params.keys()])
                action_loss=action_loss+ewc_loss

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2 / cfg1.timesteps


            if timestep % cfg1.eval_interval == 0:
                eval_mean, eval_std = eval(eval_env, model, rtg,env_index)
                utils.write_to_dict(local_log_dict, 'eval_steps', timestep - 1, using_mp)
                utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp)
                d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)
                utils.write_to_dict(local_log_dict, 'd4rl_score', d4rl_score, using_mp)
                logger.info(f"Seed: {seed}, Env: {env_name}, Step: {timestep}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

                if eval_mean > best_reward:
                    best_reward = eval_mean
                    model.save(f'{env_index}_best_model_seed_{seed}')
                    logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {timestep}')

        fisher_info_list.append(fisher_info)


        model.save(f'{env_index}_final_model_seed_{seed}')

    logger.info(f"Finish training seed {seed} with average eval mean: {eval_mean}")

    return eval_mean


def final_eval(env_list, rtg_target,logger,cfg,seed,cmd,exp_path="",num=2):
    # parallel evaluation with vectorized environment
    logger.info(f'--------------------forgetting test-------------------')

    model_dict = torch.load(exp_path+f'models/{num}_best_model_seed_{seed}.pt')


    for i in range(len(env_list)):
        env_name=env_list[i]
        eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes,
                               asynchronous=False, wrappers=RecordEpisodeStatistics)
        utils.set_seed_everywhere(eval_env, seed)

        state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
        action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
        buffer=SequenceBuffer(env_name,"medium",cfg.buffer.context_len,cmd,1,"traj_length",cfg.buffer.pos_encoding,seed)
        model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim,
                            action_space=eval_env.envs[0].action_space, state_mean=buffer.state_mean,
                            state_std=buffer.state_std, device=device)
        task_model_dict = torch.load(exp_path+f'models/{i}_final_model_seed_{seed}.pt')
        model.load_state_dict(task_model_dict)
        dic1 = model.state_dict()
        for key, var in model_dict.items():
            if "block" in key:
                dic1[key] = var
        model.load_state_dict(dic1)
        rtg=rtg_target[i]
        eval_mean, eval_std=eval(eval_env,model,rtg,env_index=i)

        logger.info(f"Seed: {seed}, env: {env_name}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

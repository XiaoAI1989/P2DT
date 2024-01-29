import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics

import utils
from buffer import SequenceBuffer
from model import DecisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def final_eval(env_list, rtg_target,logger,cfg,seed,cmd,root_dir,num=2):
    # parallel evaluation with vectorized environment
    logger.info(f'--------------------forgetting test-------------------')


    model_dict = torch.load(cmd+f'models/{num}_final_model_seed_{seed}.pt')


    for i in range(len(env_list)):
        env_name=env_list[i]
        env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes,
                               asynchronous=False, wrappers=RecordEpisodeStatistics)
        utils.set_seed_everywhere(env, seed)

        state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
        action_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
        buffer=SequenceBuffer(env_name,"medium",cfg.buffer.context_len,root_dir,1,"traj_length",cfg.buffer.pos_encoding,seed)
        model=DecisionTransformer(state_dim,action_dim,1,3,128,20,0.1,max_timestep= 1000, reward_scale= 1000,action_space=env.envs[0].action_space,state_mean=buffer.state_mean, state_std=buffer.state_std, device=device,prompt_len=20)
        dic1 = model.state_dict()
        for key, var in model_dict.items():
            if "block" in key:
                dic1[key] = var
        model.load_state_dict(dic1)
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

        reward_to_go, timestep = rtg_target[i], 0

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
                                                                  task_id=num)

            action = action_preds[:, -1]
            actions[:, timestep - 1] = action
            action = action.detach().cpu().numpy()
            state, reward, done, truncated, _ = env.step(action)

            done_flags = (1 - (1 - done) * (1 - truncated))
            reward = reward * (1 - done_flags)
            returns += reward
            reward = reward.reshape(10, 1)
            reward_to_go -= reward
            torch.cuda.empty_cache()
            if timestep >= max_timestep:
                break
        logger.info(f"Seed: {seed}, env: {env_name}, Eval mean: {np.mean(returns):.2f}, Eval std: {np.std(returns):.2f}")

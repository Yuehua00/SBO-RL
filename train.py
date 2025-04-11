import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from config import args

def train_critic(critic: Critic, critic_optimizer: torch.optim, critic_target: Critic, actor_target: Actor, replay_buffer: ReplayBuffer):

    states, actions, next_states, rewards, not_dones = replay_buffer.sample(args.batch_size)
    with torch.no_grad():
        noise = (torch.randn_like(actions) * args.policy_noise).clamp(-args.policy_noise_clip, args.policy_noise_clip)
        # print(f"noise: {noise.shape}")
        # print(f"NS: {next_states.shape}")
        # print(f"A: {actor_target(next_states).shape}")
        next_actions = (actor_target(next_states) + noise).clamp(-actor_target.max_action, actor_target.max_action)

        target_q1, target_q2 = critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + not_dones * args.gamma * target_q

    q1, q2 = critic(states, actions)

    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # 更新 target
    with torch.no_grad():
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


def train_actor(actor: Actor, actor_optimizer: torch.optim ,critic: Critic, replay_buffer: ReplayBuffer):

    states, _ = replay_buffer.sample_sa(args.batch_size)

    actor_actions = actor(states)
    actor_loss = -critic.forward_Q1(states, actor_actions).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
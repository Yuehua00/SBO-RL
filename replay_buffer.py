import numpy as np
import torch

from config import args
from torch import Tensor


class ReplayBuffer:
    
    def __init__(self, state_dim: int, action_dim: int):
        self.ptr: int = 0   # ReplayBuffer的pointer，代表現在要寫入的位置
        self.current_size: int = 0
        
        self.states: Tensor = torch.zeros(size = (args.replay_buffer_size, state_dim), dtype = torch.float32).to(args.device)
        self.actions: Tensor = torch.zeros(size = (args.replay_buffer_size, action_dim), dtype = torch.float32).to(args.device)
        self.next_states: Tensor = torch.zeros(size = (args.replay_buffer_size, state_dim), dtype = torch.float32).to(args.device)
        self.rewards: Tensor = torch.zeros(size = (args.replay_buffer_size, 1), dtype = torch.float32).to(args.device)
        self.not_dones: Tensor = torch.zeros(size = (args.replay_buffer_size, 1), dtype = torch.float32).to(args.device)


    def push(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, not_done: bool):
        state = torch.from_numpy(state).to(torch.float32).detach().to(args.device)
        action = torch.from_numpy(action).to(torch.float32).detach().to(args.device)
        next_state = torch.from_numpy(next_state).to(torch.float32).detach().to(args.device)
        reward = torch.tensor([reward]).to(torch.float32).detach().to(args.device)
        not_done = torch.tensor([not_done]).to(torch.float32).detach().to(args.device)
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = not_done
        
        self.ptr = (self.ptr + 1) % args.replay_buffer_size
        self.current_size = min(self.current_size + 1, args.replay_buffer_size)
        

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.current_size, size = batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.not_dones[indices]
        )
        
    def sample_sa(self, batch_size: int):
        indices = np.random.randint(0, self.current_size, size = batch_size)
        return (
            self.states[indices],
            self.actions[indices],
        )
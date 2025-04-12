import copy
import torch
import numpy as np
import gymnasium as gym
from copy import deepcopy

from config import args
from model import Actor, Critic
from EA.CEM import CEM
from EA.GA import GA
from learning_curve import LearningCurve
from replay_buffer import ReplayBuffer


class Task:
    def __init__(self, task_name: str):
        
        self.env = gym.make(task_name)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = torch.tensor(self.env.action_space.high).detach().to(args.device)
        
        self.cem = CEM()
        self.ga = GA()

        # actor
        self.mu_actor = Actor(self.state_dim, self.action_dim, self.max_action).to(args.device)
        self.actor = []
        # critic
        self.critic = Critic(self.state_dim, self.action_dim).to(args.device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_learning_rate)

        # 從哪裡轉換來
        self.transfer_from = None

        self.steps = 0
        
        # 上一次evaluation花多少步
        self.evaluate_steps = 0
        
        # 用於固定random seed
        self.env.reset(seed=args.seed)
        self.env.action_space.seed(args.seed)
    
    def is_reach_start_steps(self):
        return self.steps >= args.start_steps
    
    def is_reach_steps_limit(self):
        return self.steps >= args.max_steps
    
    def init_actor(self):
        self.actor = self.cem.get_init_actor_population(self.mu_actor)
    
    def evaluate(self, episodes: int, actor: Actor, replay_buffer: ReplayBuffer, learning_curve: LearningCurve):
        with torch.no_grad():
            scores: list[float] = []
            evaluate_steps = 0
            for e in range(episodes):
                state, _ = self.env.reset()
                done = False
                reach_steps_limit = False
                episode_score = 0
                
                while (not (done or reach_steps_limit)):
                    self.steps += 1
                    evaluate_steps += 1
                    learning_curve.add_step()
                    
                    state_tensor = torch.from_numpy(state).to(torch.float32).to(args.device)
                    action = actor(state_tensor).cpu().numpy()
                    next_state, reward, done, reach_steps_limit, _ = self.env.step(action)
                    episode_score += reward
                    replay_buffer.push(state, action, next_state, reward, not done)
                    state = next_state
                scores.append(episode_score)
        return np.mean(scores), evaluate_steps
    
    def survive(self, population: list[Actor], offspring: list[Actor]):
        # mu_and_lambda = population + offspring
        # mu_and_lambda.sort(key = lambda actor: actor.fitness, reverse = True)
        # # (mu + lambda) surivivial selection
        # population[:] = mu_and_lambda[:args.population_size]
        # (mu, lambda)
        population[:] = copy.deepcopy(offspring)
        population.sort(key = lambda actor: actor.fitness, reverse = True)
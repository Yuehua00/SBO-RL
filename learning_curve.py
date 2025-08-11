import gymnasium as gym
import torch

import os
import json
import datetime
import random
import string
from copy import deepcopy
from EA.EA_utils import gene_to_phene


from config import args
from model import Actor


class LearningCurve:

    def __init__(self, env_name: str, model_actor: Actor, mu_actor_gene: list, network_size: list, max_network_size: list):

        self.steps = 0
        self.learning_curve_steps = []
        self.learning_curve_scores = []
        self.learning_curve_mu_actor = []
        self.transfer_size = []
        self.transfer_from = []

        self.network_size = network_size
        self.max_network_size = max_network_size

        self.env_name = env_name
        self.mu_actor = gene_to_phene(deepcopy(model_actor), mu_actor_gene, self.network_size, self.max_network_size).to(args.device)

        self.test_initial_performance()


    def update(self, mu_actor_gene: list):
        self.mu_actor = gene_to_phene(self.mu_actor, mu_actor_gene, self.network_size, self.max_network_size).to(args.device)


    def test_initial_performance(self):

        self.learning_curve_steps.append(0)
        self.learning_curve_scores.append(self.test_performance(self.env_name, self.mu_actor))


    def add_step(self):

        self.steps += 1

        if ((self.steps % args.test_performance_freq == 0) and (self.steps <= args.max_steps)):

            self.learning_curve_steps.append(self.steps)

            # [Debug]
            # print(self.steps)

            if self.steps > args.start_steps:
                score = self.test_performance(self.env_name, self.mu_actor)
                self.learning_curve_scores.append(score)
            else:
                self.learning_curve_scores.append(self.learning_curve_scores[-1])


    def test_performance(self, env_name: str, actor: Actor):

        env = gym.make(env_name)
        env.reset(seed = args.seed + 555)

        avg_score = 0.

        with torch.no_grad():

            for t in range(args.test_n):

                state , _ = env.reset()

                done = False
                reach_step_limit = False

                while (not done) and (not reach_step_limit):

                    state = torch.from_numpy(state).to(torch.float32).to(device=args.device)

                    action = actor(state)
                    action = action.cpu().numpy()

                    state , reward , done , reach_step_limit , _ = env.step(action)

                    avg_score += reward

        avg_score = avg_score / args.test_n

        return avg_score


    def save(self, path: str):

        os.makedirs(args.output_path, exist_ok=True)

        file_name = f"[{args.algorithm}][{args.env_name}][{args.seed}][{datetime.date.today()}][Learning Curve][{''.join(random.choices(string.ascii_uppercase, k=6))}].json"
        path = os.path.join(args.output_path, file_name)

        with open(path, "w") as file:
            json_data = {
                "Config": vars(args),
                "Learning Curve": {
                    "Steps": self.learning_curve_steps,
                    "Score": self.learning_curve_scores
                },
                "Transfer Size": self.transfer_size,
                "Transfer From": self.transfer_from
            }

            json.dump(json_data, file)

            file.close()




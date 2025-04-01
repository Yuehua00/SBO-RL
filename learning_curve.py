import gymnasium as gym
import torch

import os
import json
import datetime
import random
import string
from copy import deepcopy


from config import args
from model import Actor


class LearningCurve:

    def __init__(self, env_name: str, mu_actor: Actor):

        self.steps = 0
        self.learning_curve_steps = []
        self.learning_curve_scores = []
        self.learning_curve_mu_actor = []

        self.env_name = env_name
        self.mu_actor = deepcopy(mu_actor)

        self.test_initial_performance()


    def update(self, mu_actor: Actor):
        self.mu_actor = deepcopy(mu_actor)


    def test_initial_performance(self):

        self.learning_curve_steps.append(0)
        self.learning_curve_scores.append(self.test_performance(self.actor))


    def add_step(self):

        self.steps += 1

        if self.steps % args.test_performance_freq == 0:

            self.learning_curve_steps.append(self.steps)

            if self.steps > args.start_steps:
                self.learning_curve_scores.append(self.test_performance(self.actor))
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

        if (not os.path.exists(args.output_path)):
            os.makedirs(args.output_path)

        # file_name = f"[{args.algorithm}][{args.env_name}][{args.seed}][{datetime.date.today()}][Learning Curve][{''.join(random.choices(string.ascii_uppercase, k=6))}].json"
        # path = os.path.join(args.output_path, file_name)

        result = {
            "Config": vars(args),
            "Learning Curve": {
                "Steps": self.learning_curve_steps,
                "Score": self.learning_curve_scores
            }
        }

        with open(path, mode="w") as file:

            json.dump(result, file)





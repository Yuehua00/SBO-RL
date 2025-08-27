import torch
import numpy as np
import json
import string
import datetime
import random
import math
import os

from config import args
from EA.EA_utils import phenes_to_genes
from model import Actor, Individual
from copy import deepcopy


class CEM_IM:

    def __init__(self):

        self.mu_actor_gene = None
        self.cov = None

        self.old_mu_gene = None
        self.old_cov = None

        self.actor_parents_size = int(args.population_size * args.CEM_parents_ratio_actor)
        self.actor_parents_weights = torch.tensor([np.log((self.actor_parents_size + 1) / i) for i in range(1, self.actor_parents_size + 1)])
        self.actor_parents_weights /= self.actor_parents_weights.sum()
        self.actor_parents_weights = self.actor_parents_weights.unsqueeze(dim=0)
        self.actor_parents_weights = self.actor_parents_weights.to(torch.float32).detach().to(args.device)

        self.damp = torch.tensor(1e-3).to(torch.float32).detach().to(args.device)
        self.damp_limit = torch.tensor(1e-5).to(torch.float32).detach().to(args.device)
        self.damp_tau = torch.tensor(0.95).to(torch.float32).detach().to(args.device)

        self.n_reused_list = []
        self.reused_idx_list = []
        self.reused_number = 0
        self.reused_idx = []


    def old_log_pdf(self, sample: list):
        
        return (
            -0.5 * torch.log(2 * torch.tensor(math.pi))
            - torch.log(self.old_cov.sqrt())
            - 0.5 * ((sample - self.old_mu_gene) / self.old_cov.sqrt()) ** 2
        ).sum()

    def new_log_pdf(self, sample: list):
        
        return (
            -0.5 * torch.log(2 * torch.tensor(math.pi))
            - torch.log(self.cov.sqrt())
            - 0.5 * ((sample - self.mu_actor_gene) / self.cov.sqrt()) ** 2
        ).sum()
    
    def sample_one_actor(self) -> Individual:

        param_size = self.mu_actor_gene.shape[0]
        epsilon = torch.randn(param_size, dtype=torch.float32, device=args.device)
        individual = (self.mu_actor_gene + epsilon * self.cov.sqrt()).view(-1)

        new_actor = Individual()
        new_actor.gene = individual
        new_actor.fitness = None
        new_actor.transfer_from = None
        new_actor.alpha = None
        
        return new_actor


    def get_init_actor_population(self, mu_actor_gene: torch.tensor) -> list[Individual]:

        self.mu_actor_gene = mu_actor_gene
        params_size = self.mu_actor_gene.shape[0]

        self.cov = (args.CEM_sigma_init) * torch.ones(params_size, dtype=torch.float32, device=args.device)

        self.old_mu_gene = self.mu_actor_gene
        self.old_cov = self.cov
        
        epsilon_half = torch.randn(args.population_size // 2, params_size, dtype=torch.float32, device=args.device)
        epsilon = torch.cat([epsilon_half, -epsilon_half], dim=0)

        new_gene = self.mu_actor_gene + epsilon * self.cov.sqrt()

        actor_population: list[Individual] = []

        for i in range(args.population_size):

            actor_tmp = Individual()
            actor_tmp.gene = new_gene[i]

            actor_population.append(actor_tmp)

        return actor_population


    def variate(self, actor_population: list[Individual], offspring_size: int):

        with torch.no_grad():

            # actor_population = deepcopy(actor_population)
            actor_population.sort(key=lambda actor: actor.fitness, reverse=True)

            # genes = phenes_to_genes(actor_population)  # shape = (population_size , params_size)
            genes = torch.cat([actor.gene.unsqueeze(0) for actor in actor_population], dim=0)   # shape = (population_size , params_size)

            self.old_mu_gene = self.mu_actor_gene  # shape = (1, params_size)
            self.mu_actor_gene = torch.matmul(self.actor_parents_weights, genes[: self.actor_parents_size]).squeeze(0)  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size) -> (params_size,)

            self.cov = genes[ : self.actor_parents_size] - self.old_mu_gene      # shape = (actor_parents_size , params_size)
            self.cov = self.cov.pow(2)                                      # shape = (actor_parents_size , params_size)
            self.cov = torch.matmul(self.actor_parents_weights , self.cov)  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size)
            self.cov += self.damp                                      # shape = (1 , params_size)

            # cov = self.actor_cov_discount * cov                   # shape = (1 , params_size)

            self.damp = self.damp_tau * self.damp + (1 - self.damp_tau) * self.damp_limit

            # importance mixing
            n_reused = 0
            n_sampled = 0

            idx_reused = []
            old_samples: list[Individual] = actor_population

            new_population: list[Individual] = [Individual() for _ in range(offspring_size)]

            individual_life = []

            for i in range(offspring_size):

                sample = old_samples[i]

                if n_reused + n_sampled < args.population_size:

                    u = np.random.uniform(0, 1)
                    if np.log(u) < self.new_log_pdf(sample.gene) - self.old_log_pdf(sample.gene):
                        
                        sample.life += 1
                        new_population[n_reused] = sample
                        idx_reused.append(i)
                        n_reused += 1
                    
                    else:
                        individual_life.append(sample.life)
                
                if n_reused + n_sampled < args.population_size:

                    new_sample = self.sample_one_actor()

                    u = np.random.uniform(0, 1)
                    if np.log(1-u) >= self.old_log_pdf(new_sample.gene) - self.new_log_pdf(new_sample.gene):
                        
                        new_population[-n_sampled-1] = new_sample
                        n_sampled += 1

                if n_reused + n_sampled >= args.population_size:

                    break

            cpt = n_reused + n_sampled
            while cpt < offspring_size:
                new_sample = self.sample_one_actor()
                new_population[cpt - n_sampled] = new_sample
                cpt += 1

            self.old_mu_gene = self.mu_actor_gene
            self.old_cov = self.cov

            self.n_reused_list.append(n_reused)
            self.reused_idx_list.append(idx_reused)
            self.reused_number = n_reused
            self.reused_idx = idx_reused

            return new_population, individual_life
        
    
    # def save(self, env_name):

    #     os.makedirs(args.output_path, exist_ok=True)

    #     file_name = f"[{args.algorithm}][{env_name}][{args.seed}][{datetime.date.today()}][Reused Information][{''.join(random.choices(string.ascii_uppercase, k=6))}].json"
    #     path = os.path.join(args.output_path, file_name)

    #     result = {
    #         "Number of reused" : self.n_reused_list,
    #         "Reused idndex" : self.reused_idx_list
    #     }

    #     with open(path, mode='w') as file:
            
    #         json.dump(result, file)
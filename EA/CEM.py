import torch
import numpy as np

from config import args
from EA.EA_utils import phenes_to_genes
from model import Actor
from copy import deepcopy


class CEM:

    def __init__(self):

        self.actor_mu = None
        self.actor_sigma = None

        self.actor_parents_size = int(args.population_size * args.CEM_parents_ratio_actor)
        self.actor_parents_weights = torch.tensor([np.log((self.actor_parents_size + 1) / i) for i in range(1 , self.actor_parents_size + 1)])
        self.actor_parents_weights /= self.actor_parents_weights.sum()
        self.actor_parents_weights = self.actor_parents_weights.unsqueeze(dim=0)
        self.actor_parents_weights = self.actor_parents_weights.to(torch.float32).detach().to(args.device)  # shape = (1 , actor_parents_size)

        self.damp = torch.tensor(1e-3).to(torch.float32).detach().to(args.device)
        self.damp_limit = torch.tensor(1e-5).to(torch.float32).detach().to(args.device)
        self.damp_tau = torch.tensor(0.95).to(torch.float32).detach().to(args.device)

        self.actor_cov_discount = torch.tensor(args.CEM_cov_discount_actor).to(torch.float32).detach().to(args.device)

    def get_init_actor_population(self, mu_actor: Actor) -> list[Actor]:
        
        genes = phenes_to_genes([mu_actor])  # shape = (1 , params_size)
        self.actor_mu = genes
        params_size = self.actor_mu.shape[1]
        
        cov = (args.CEM_sigma_init) * torch.ones(params_size, dtype = torch.float32, device = args.device)
        
        epsilon = torch.randn(args.population_size, params_size, dtype = torch.float32, device = args.device)    # shape = (population_size , params_size)

        new_genes = self.actor_mu + epsilon  * cov.sqrt() # shape = (population_size , params_size)

        actor_population: list[Actor] = []

        for i in range(args.population_size):

            ctr = 0

            actor = deepcopy(mu_actor)
            for param in actor.parameters():

                n = param.numel()

                param.data.copy_(new_genes[i , ctr : ctr + n].view(param.shape))

                ctr += n
            actor_population.append(actor)
        return actor_population

    def variate(self, actor_population: list[Actor], offspring_size: int) -> list[Actor]:
        print("actor_mu.shape =", self.actor_mu.shape)

        with torch.no_grad():

            actor_population = deepcopy(actor_population)
            actor_population.sort(key = lambda actor: actor.fitness, reverse = True)

            genes = phenes_to_genes(actor_population)  # shape = (population_size , params_size)


            old_mu = self.actor_mu

            self.actor_mu = torch.matmul(self.actor_parents_weights , genes[ : self.actor_parents_size])  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size)


            cov = genes[ : self.actor_parents_size] - old_mu      # shape = (actor_parents_size , params_size)
            cov = cov.pow(2)                                      # shape = (actor_parents_size , params_size)
            cov = torch.matmul(self.actor_parents_weights , cov)  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size)
            cov += self.damp                                      # shape = (1 , params_size)

            cov = self.actor_cov_discount * cov                   # shape = (1 , params_size)


            self.damp = self.damp_tau * self.damp + (1 - self.damp_tau) * self.damp_limit


            epsilon = torch.randn_like(genes)    # shape = (population_size , params_size)

            new_genes = self.actor_mu + epsilon * cov.sqrt()  # shape = (population_size , params_size)

            print("mu diff:", (self.actor_mu - old_mu).abs().mean().item())

            for i in range(offspring_size):

                ctr = 0

                for param in actor_population[i].parameters():

                    n = param.numel()

                    param.data.copy_(new_genes[i , ctr : ctr + n].view(param.shape))

                    ctr += n


        return actor_population[:offspring_size]
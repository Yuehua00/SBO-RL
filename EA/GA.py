import torch
import torch.nn as nn

from model import Individual

class GA:

    def __init__(self):
        pass

    # def phene_to_gene(self, individual: nn.Module) -> torch.Tensor:

    #     collect_gene = []

    #     for param in individual.parameters():
    #         collect_gene.append(param.view(-1))

    #     gene = torch.cat(collect_gene, dim=0)
    #     gene = gene.clone().detach()

    #     return gene
            

    # def gene_to_phene(self, individual: nn.Module, gene) -> nn.Module:
        
    #     ctr = 0

    #     for param in individual.parameters():
    #         n = param.numel()
    #         param.data.copy_(gene[ctr:ctr+n].view(param.shape))
    #         ctr += n

    #     return individual


    # def transfer(self, task_i_actors: list[nn.Module], task_j_actors: list[nn.Module], transfer_size) -> list[nn.Module]:
        
    #     offsprings: list[nn.Module] = []

    #     with torch.no_grad():
    #         for index in range(len(task_i_actors)):
    #             # 要做交換的
    #             if index >= transfer_size-1:
    #                 parent1 = deepcopy(task_i_actors[index])
    #                 parent2 = deepcopy(task_j_actors[index])
    #                 parent1 = self.phene_to_gene(parent1)
    #                 parent2 = self.phene_to_gene(parent2)

    #                 len_p1 = len(parent1)
    #                 len_p2 = len(parent2)

    #                 # 長度足夠
    #                 if len_p2 >= len_p1:
    #                     offspring = parent2[:len_p1]
    #                 # 長度不夠
    #                 else :
    #                     offspring = torch.cat([parent2, parent1[len_p2:]], dim=0)

    #                 offspring = self.gene_to_phene(task_i_actors[index], offspring)

    #             # 維持原本的
    #             else:
    #                 offspring = task_i_actors[index]

    #             offsprings.append(offspring)

    #     return offsprings

    def transfer(self, task_i_actors: list[Individual], task_j_actors: list[Individual], transfer_size: int) -> list[Individual]:

        task_i_actors[-transfer_size:] = task_j_actors[-transfer_size:]

        for actor in task_i_actors[-transfer_size:]:
            actor.fitness = None

        return task_i_actors
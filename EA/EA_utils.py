import torch
import torch.nn as nn


def phenes_to_genes(population: list[nn.Module]) -> torch.Tensor:

    genes = []

    for individual in population:

        genes.append(torch.cat([param.view(-1) for param in individual.parameters()]))
    
    genes = torch.stack(genes)  # shape = (population_size , params_size)
    genes = genes.detach()

    return genes


def phene_to_gene(individual: nn.Module) -> torch.Tensor:

    # 全部大小，攤成一維
    collect_gene = torch.cat([param.reshape(-1) if param.dim() > 1 else param for param in individual.parameters()]).detach()

    return collect_gene


def gene_to_phene(individual: nn.Module, gene: torch.Tensor, network_size, max_network_size) -> nn.Module:
    ctr = 0
    for param, (input_size, output_size), (max_input_size, max_output_size) in zip(individual.parameters(), network_size, max_network_size):
        # 用來暫存該層參數
        param_list = []
        for i in range(input_size):
            start = ctr + i * max_output_size
            row_gene = gene[start : start + output_size]
            param_list.append(row_gene)
        
        collect_gene = torch.cat(param_list)
        param.data.copy_(collect_gene.view(param.shape))
        
        ctr += max_input_size * max_output_size

    return individual


def local_phene_to_gene(individual: nn.Module, gene: torch.Tensor, network_size, max_network_size):
    ctr = 0
    for param, (input_size, output_size), (max_input_size, max_output_size) in zip(
        individual.parameters(), network_size, max_network_size
    ):
        param_data = param.data.view(input_size, output_size)

        for i in range(input_size):
            start = ctr + i * max_output_size
            end = start + output_size
            gene[start:end] = param_data[i]  # 只覆蓋有效部分

        ctr += max_input_size * max_output_size  # 跳到下一層的全域段落

    return gene

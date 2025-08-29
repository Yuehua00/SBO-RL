import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
import os
from datetime import datetime
import time


from config import args
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve
from Task import Task
from train import train_critic, train_actor
from EA.EA_utils import gene_to_phene, phene_to_gene, local_phene_to_gene
from EA.GA import GA


if (__name__ == "__main__"):

    start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ###### 設定隨機種子 ######
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###### 建立環境 ######
    tasks = [Task(env_name) for env_name in args.env_names]

    # 找到最大維度
    max_state_dim = max([task.state_dim for task in tasks])
    max_action_dim = max([task.action_dim for task in tasks])

    max_network_size = [
        (400, max_state_dim),
        (1, 400),
        (300, 400),
        (1, 300),
        (max_action_dim, 300),
        (1, max_action_dim)
    ]
    for task in tasks:
        task.max_network_size = max_network_size

    # 初始化 replay buffer
    replay_buffers = [ReplayBuffer(task.state_dim, task.action_dim) for task in tasks]


    ###### framework 流程 ######

    ###### 初始化 Symbiosis estimation ######
    mutualism = np.ones(shape=(len(tasks), len(tasks)), dtype=np.int32)
    neutralism = np.ones(shape=(len(tasks), len(tasks)), dtype=np.int32)
    competition = np.ones(shape=(len(tasks), len(tasks)), dtype=np.int32)
    commensalism = np.ones(shape=(len(tasks), len(tasks)), dtype=np.int32)
    amensalism = np.ones(shape=(len(tasks), len(tasks)), dtype=np.int32)
    parasitism = np.ones(shape=(len(tasks), len(tasks)), dtype=np.int32)

    # 初始化轉移率
    # rate_transfer = np.zeros((len(tasks), len(tasks)))
    # 初始化轉移數量
    adapt_transfer_size = np.zeros((len(tasks), len(tasks)), dtype = np.int32)

    ga = GA()

    # 每個任務的初始 actor
    for task in tasks:
        task.init_mu_actor(max_state_dim, max_action_dim)
        task.init_actors()

    ###### 初始化 learning curve ######
    learning_curves=[
        LearningCurve(args.env_names[task_i], tasks[task_i].model_actor, tasks[task_i].mu_actor_gene, tasks[task_i].network_size, max_network_size, task_i) for task_i in range(len(tasks))
    ]

    ###### 最一開始的 population 計算 fitness ######
    for task_i in range(len(tasks)):
        task = tasks[task_i]
        task.evaluate_steps = 0

        for actor in task.actors:
            fitness, evaluate_steps = task.evaluate(1, actor, replay_buffers[task_i], learning_curves[task_i])
            # print(f"Actor fitness = {fitness}")
            actor.fitness = fitness
            task.evaluate_steps += evaluate_steps

        # [Debug]
        avg_fitness = 0
        for i, actor in enumerate(tasks[task_i].actors):
            print(f"Actor {i} fitness: {actor.fitness: .4f}")
            avg_fitness += actor.fitness
        avg_fitness /= len(tasks[task_i].actors)
        print("=============================================")
        print(f"Task [{args.env_names[task_i]}] avg fitness: {avg_fitness: .4f}")
        print("=============================================")


    ###### 填充replay buffer 階段(用CEM演化去填充) ######
    all_reach_start_steps = False
    while(not all_reach_start_steps):

        for task_i in range(len(tasks)):

            if (tasks[task_i].is_reach_start_steps()):
                print(f"Task [{args.env_names[task_i]}] is ready to train.")
                continue
        
            # CEM 抽新的 offsprings
            tasks[task_i].actors, individual_life = tasks[task_i].cem.variate(tasks[task_i].actors, args.population_size)
            learning_curves[task_i].individual_life = individual_life

            # 更新 learning curve 裡的 mu actor
            tasks[task_i].mu_actor_gene = tasks[task_i].cem.mu_actor_gene
            learning_curves[task_i].update(tasks[task_i].mu_actor_gene, tasks[task_i].cem.reused_number, tasks[task_i].cem.reused_idx)
            
            ###### 評估所有 offsprings 的 actor ######
            tasks[task_i].evaluate_steps = 0

            print(f"Task [{args.env_names[task_i]}] evaluate:")
            print(f"Current steps: {tasks[task_i].steps}")

            for actor in tasks[task_i].actors:
                fitness, evaluate_steps = tasks[task_i].evaluate(1, actor, replay_buffers[task_i], learning_curves[task_i])
                actor.fitness = fitness 
                tasks[task_i].evaluate_steps += evaluate_steps

            # [Debug]
            avg_fitness = 0
            for i, actor in enumerate(tasks[task_i].actors):
                print(f"Actor {i} fitness: {actor.fitness: .4f}")
                avg_fitness += actor.fitness
            avg_fitness /= len(tasks[task_i].actors)
            print("=============================================")
            print(f"Task [{args.env_names[task_i]}] avg fitness: {avg_fitness: .4f}")
            print("=============================================")

        all_reach_start_steps = True
        for task in tasks:
            all_reach_start_steps = all_reach_start_steps and task.is_reach_start_steps()


    ###### 主循環 ######
    all_reach_max_steps = False
    indv_ranking: list[np.ndarray] = [None for _ in range(len(tasks))]
    while(not all_reach_max_steps):
        
        for task_i in range(len(tasks)):

            if (tasks[task_i].is_reach_steps_limit()):
                print(f"Task [{args.env_names[task_i]}] is frozen.")
                continue
            
            print("=============================================")
            print(f"Task [{args.env_names[task_i]}] is doing policy gradient...")
            print("=============================================")

            # 重新選取 actor population
            tasks[task_i].actors, individual_life = tasks[task_i].cem.variate(tasks[task_i].actors, args.population_size)
            for x in tasks[task_i].actors:
                x.transfer_from = None
            learning_curves[task_i].individual_life = individual_life

            # train
            print(f"Task [{args.env_names[task_i]}] train:")

            critic = tasks[task_i].critic
            critic_optimizer = tasks[task_i].critic_optimizer
            critic_target = tasks[task_i].critic_target

            replay_buffer = replay_buffers[task_i]
            
            half_size = len(tasks[task_i].actors) // 2
            for i in range(half_size):

                actor = gene_to_phene(tasks[task_i].model_actor, tasks[task_i].actors[i].gene, tasks[task_i].network_size, tasks[task_i].max_network_size)
                actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
                # actor_target = deepcopy(actor).requires_grad_(False)

                for _ in range(tasks[task_i].evaluate_steps // half_size):
                    train_critic(critic, critic_optimizer, critic_target, actor, replay_buffer)
                
                for _ in range(tasks[task_i].evaluate_steps):
                    train_actor(actor, actor_optimizer, critic, replay_buffer)

                local_phene_to_gene(actor, tasks[task_i].actors[i].gene, tasks[task_i].network_size, tasks[task_i].max_network_size)

            # 更新 learning curve 裡的 mu actor
            tasks[task_i].mu_actor_gene = tasks[task_i].cem.mu_actor_gene  # tasks[task_i].mu_actor[0] 是甚麼
            learning_curves[task_i].update(tasks[task_i].mu_actor_gene, tasks[task_i].cem.reused_number, tasks[task_i].cem.reused_idx)
        
        for task_i in range(len(tasks)):

            if (tasks[task_i].is_reach_steps_limit()):
                print(f"Task [{args.env_names[task_i]}] is frozen.")
                continue

            print("=============================================")
            print(f"Task [{args.env_names[task_i]}] is doing Transfer...")
            print("=============================================")

            task_j = 0
            for j in range(len(tasks)):
                
                if task_i == j:
                    # rate_transfer[task_i][j] = 0
                    continue
                
                r = 0
                # 計算 transfer rata
                positive = mutualism[task_i][j] + commensalism[task_i][j] + parasitism[task_i][j]
                negative = neutralism[task_i][j] + amensalism[task_i][j] + competition[task_i][j]

                r_j = positive / (positive + negative + 1e-15)

                if r_j > 0.5:
                    r_j = 0.5

                if r < r_j:
                    r = r_j
                    task_j = j
                # rate_transfer[task_i][j] = r_j
            print(r)
            print("----------------------------------------")
            tasks[task_i].transfer_from = None
            transfer_record = [0 for _ in range(len(args.env_names))]
            if r >= np.random.uniform(0, 1):
                lambda_i = len(tasks[task_i].actors)
                s = math.floor(r * lambda_i)  # s = r * args.population_size
                if s < 1:
                    s = 1
                learning_curves[task_i].once_transfer_size = s
                # 紀錄多少轉換
                adapt_transfer_size[task_i][task_j] = s
                # 從哪裡轉換
                tasks[task_i].transfer_from = task_j
                transfer_record[task_j] = s
                # 轉換(要轉換的、要轉換過去的、轉換數量)
                ga.transfer(tasks[task_i].actors, tasks[task_j].actors, s, task_j)
                # tasks[task_i].actors[lambda_i-s :] = tasks[task_j].actor[ : s]
            else:
                learning_curves[task_i].once_transfer_size = 0

            for i, number in enumerate(transfer_record):
                learning_curves[task_i].once_transfer_record[i] = number

            print(f"Task [{args.env_names[task_i]}]")
            print(f"transfer from task [{args.env_names[task_j]}] with size {learning_curves[task_i].once_transfer_size}")
            print(f"{transfer_record}")


            ###### Evaluate ######
            tasks[task_i].evaluate_steps = 0
            print(f"Task [{args.env_names[task_i]}] evaluate:")
            print(f"Current steps: {tasks[task_i].steps}")
            print(f"Score: {learning_curves[task_i].learning_curve_scores[-1]}")


            for individual in tasks[task_i].actors:
                # actor = gene_to_phene(tasks[task_i].model_actor, individual.gene, tasks[task_i].network_size, tasks[task_i].max_network_size)
                fitness, evaluate_steps = tasks[task_i].evaluate(1, individual, replay_buffers[task_i], learning_curves[task_i])
                individual.fitness = fitness
                tasks[task_i].evaluate_steps += evaluate_steps

            transfer_in_five_num = 0
            if (not tasks[task_i].is_reach_steps_limit()):
                tasks[task_i].actors.sort(key = lambda actor: actor.fitness, reverse = True)

                # [Debug]
                avg_fitness = 0
                for i, actor in enumerate(tasks[task_i].actors):
                    print(f"Actor {i} fitness: {actor.fitness} transfer from {actor.transfer_from}")
                    avg_fitness += actor.fitness

                    # 紀錄交流來個體中，有多少是在前五名內
                    if (actor.transfer_from is not None) and (actor.transfer_from != task_i) and (i < 5):
                        transfer_in_five_num +=1

                avg_fitness /= len(tasks[task_i].actors)
                print("=============================================")
                print(f"Task [{args.env_names[task_i]}] avg fitness: {avg_fitness: .4f}")
                print("=============================================")
            else:
                print(f"Task [{args.env_names[task_i]}] is frozen.")

            learning_curves[task_i].transfer_in_five_num = transfer_in_five_num

            # 排序(轉換過來的和原本的，由大到小) 
            # indv_ranking[task_i] = np.argsort(-tasks[task_i].actors)
            fitness_arr = np.array([actor.fitness for actor in tasks[task_i].actors])
            indv_ranking[task_i] = np.argsort(np.argsort(-fitness_arr))

        # update_Symbiosis_estimation
        ratio_help_neutral_harm = [0.25, 0.5, 1.0]

        # 更新之間關係 (九種可能性)
        for task_i in range(len(tasks)):

            if tasks[task_i].is_reach_steps_limit():
                print(f"Task [{args.env_names[task_i]}] is frozen.")
                continue

            print(f"Task [{args.env_names[task_i]}] is updating relations...")

            actor_size = len(tasks[task_i].actors)

            help_pos = int(ratio_help_neutral_harm[0] * actor_size)
            neutral_pos = int(ratio_help_neutral_harm[1] * actor_size)
            harm_pos = int(ratio_help_neutral_harm[2] * actor_size)

            # -------------------------------
            # Help 區間
            # -------------------------------
            for index in range(help_pos):
                task_j = tasks[task_i].actors[index].transfer_from  # 個體的來源任務

                # 沒有轉換
                if task_j is None:
                    mutualism[task_i][task_i] += 1
                    continue

                transfer_pos = actor_size - adapt_transfer_size[task_i][task_j]
                if index >= transfer_pos:  # 轉換來的
                    src_rank = indv_ranking[task_j][index]  # 在來源 task 的排名

                    if src_rank < help_pos:
                        mutualism[task_i][task_j] += 1
                    elif src_rank < neutral_pos:
                        commensalism[task_i][task_j] += 1
                    else:
                        parasitism[task_i][task_j] += 1
                else:  # 自己 task
                    mutualism[task_i][task_i] += 1

            # -------------------------------
            # Neutral 區間
            # -------------------------------
            for index in range(help_pos, neutral_pos):
                task_j = tasks[task_i].actors[index].transfer_from

                if task_j is None:
                    neutralism[task_i][task_i] += 1
                    continue

                transfer_pos = actor_size - adapt_transfer_size[task_i][task_j]
                if index >= transfer_pos:
                    src_rank = indv_ranking[task_j][index]

                    if src_rank < help_pos:
                        pass  # 不計
                    elif src_rank < neutral_pos:
                        neutralism[task_i][task_j] += 1
                    else:
                        amensalism[task_i][task_j] += 1
                else:
                    neutralism[task_i][task_i] += 1

            # -------------------------------
            # Harm 區間
            # -------------------------------
            for index in range(neutral_pos, harm_pos):
                task_j = tasks[task_i].actors[index].transfer_from

                if task_j is None:
                    competition[task_i][task_i] += 1
                    continue

                transfer_pos = actor_size - adapt_transfer_size[task_i][task_j]
                if index >= transfer_pos:
                    src_rank = indv_ranking[task_j][index]

                    if src_rank < help_pos:
                        pass
                    elif src_rank < neutral_pos:
                        pass
                    else:
                        competition[task_i][task_j] += 1
                else:
                    competition[task_i][task_i] += 1


        # 檢查是否已經都達到 max steps 了
        all_reach_max_steps = True
        for task in tasks:
            all_reach_max_steps = all_reach_max_steps and task.is_reach_steps_limit()



    ###### 儲存結果 ######
    if args.save_result == True:
        for i in range(len(tasks)):
            learning_curves[i].save(os.path.join(args.output_path, f"[{args.env_names[i]}]Learning Curve.json"))

    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    if (args.save_result):
        with open(os.path.join(args.output_path, f"{args.output_path}[{args.seed}][info].txt"), mode = "w") as file:
            file.write(f"Start date: {start_date}\n")
            file.write(f"End date: {end_date}\n")
            file.close()


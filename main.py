import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import os
from datetime import datetime
import time


from config import args
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve
from Task import Task
from train import train_critic, train_actor
from EA.EA_utils import gene_to_phene
from EA.GA import GA


if (__name__ == "__main__"):

    start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ###### 設定隨機種子 ######
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###### 建立環境 ######
    tasks = [Task(env_name) for env_name in args.env_names]

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

    ##genes_dim = 256 * 256 + 256 # 第二層的 weight + bias

    # 初始化轉移率
    rate_transfer = np.zeros((len(tasks), len(tasks)))
    # 初始化轉移數量
    adapt_transfer_size = np.zeros((len(tasks), len(tasks)))

    ga = GA()

    # 每個任務的初始 actor
    for task_i in range(len(tasks)):
        tasks[task_i].init_actor()

    ###### 初始化 learning curve ######
    learning_curves=[
        LearningCurve(args.env_names[task_i], tasks[task_i].mu_actor) for task_i in range(len(tasks))
    ]

    # 紀錄最初始的分數
    for i in range(len(learning_curves)):
        learning_curves[i].test_initial_performance()

    ###### 最一開始的 population 計算 fitness ######
    for task_i in range(len(tasks)):
        task = tasks[task_i]
        task.evaluate_steps = 0

        for i in range(len(task.actor)):
            actor = task.actor[i]
            fitness, evaluate_steps = task.evaluate(1, actor, replay_buffers[task_i], learning_curves[task_i])
            actor.fitness = fitness
            task.evaluate_steps += evaluate_steps

        # [Debug]
        avg_fitness = 0
        for i, actor in enumerate(tasks[task_i].actor):
            print(f"Actor {i} fitness: {actor.fitness: .4f}")
            avg_fitness += actor.fitness
        avg_fitness /= len(tasks[task_i].actor)
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
        
            # CEM 抽新的 offspring
            offspring = tasks[task_i].cem.variate(tasks[task_i].actor, args.population_size)

            # 更新 learning curve 裡的 mu actor
            tasks[task_i].mu_actor = gene_to_phene(tasks[task_i].mu_actor, tasks[task_i].cem.actor_mu[0])  # tasks[task_i].mu_actor[0] 是甚麼
            learning_curves[task_i].update(tasks[task_i].mu_actor)

            ###### 評估所有 offspring 的 actor ######
            tasks[task_i].evaluate_steps = 0

            print(f"Task [{args.env_names[task_i]}] evaluate:")
            print(f"Current steps: {tasks[task_i].steps}")

            for i in range(len(offspring)):
                actor = offspring[i]
                fitness, evaluate_steps = tasks[task_i].evaluate(1, actor, replay_buffers[task_i], learning_curves[task_i])
                actor.fitness = fitness 
                tasks[task_i].evaluate_steps += evaluate_steps

            tasks[task_i].survive(tasks[task_i].actor, offspring)

            # [Debug]
            avg_fitness = 0
            for i, actor in enumerate(tasks[task_i].actor):
                print(f"Actor {i} fitness: {actor.fitness: .4f}")
                avg_fitness += actor.fitness
            avg_fitness /= len(tasks[task_i].actor)
            print("=============================================")
            print(f"Task [{args.env_names[task_i]}] avg fitness: {avg_fitness: .4f}")
            print("=============================================")

            all_reach_start_steps = True
            for task in tasks:
                all_reach_start_steps = all_reach_start_steps and task.is_reach_start_steps()


    ###### 主循環 ######
    all_reach_max_steps = False
    while(not all_reach_max_steps):
        
        for task_i in range(len(tasks)):

            if (tasks[task_i].is_reach_steps_limit()):
                print(f"Task [{args.env_names[task_i]}] is frozen.")
                continue
            
            print("=============================================")
            print(f"Task [{args.env_names[task_i]}] is doing policy gradient...")
            print("=============================================")

            # 重新選取 actor population
            tasks[task_i].actor = tasks[task_i].cem.variate(tasks[task_i].actor, args.population_size)
                
            # train
            print(f"Task [{args.env_names[task_i]}] train:")
            half_size = len(tasks[task_i].actor)//2
            for i in range(half_size):

                actor = tasks[task_i].actor[i]
                actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
                actor_target = deepcopy(actor).requires_grad_(False)
                
                critic = tasks[task_i].critic
                critic_optimizer = tasks[task_i].critic_optimizer
                critic_target = tasks[task_i].critic_target

                replay_buffer = replay_buffers[task_i]

                for _ in range(tasks[task_i].evaluate_steps // half_size):
                    train_critic(critic, critic_optimizer, critic_target, actor_target, replay_buffer)
                
                for _ in range(tasks[task_i].evaluate_steps):
                    train_actor(actor, actor_optimizer, critic, replay_buffer)

        indv_ranking: list[list] = [[] for _ in range(tasks)]
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
                    rate_transfer[task_i][j] = 0
                    continue

                # 計算 transfer rata
                positive = mutualism[task_i][j] + commensalism[task_i][j] + parasitism[task_i][j]
                negative = neutralism[task_i][j] + amensalism[task_i][j] + competition[task_i][j]
                r_j = positive / (positive + negative + 1e-15)
                if r_j > 0.5:  # 兩段 code 當中都有，但 psuedo code 沒提到
                    r_j = 0.5
                if rate_transfer[task_i][j] < r_j:
                    r = r_j
                    task_j = j
                rate_transfer[task_i][j] = r_j
            if r >= np.random.uniform(0, 1, 1):
                lambda_i = len(tasks[task_i].actor)
                s = r * lambda_i  # s = r * args.population_size
                if s < 1:
                    s = 1
                # 紀錄多少轉換
                adapt_transfer_size[task_i][task_j] = s
                # 從哪裡轉換
                tasks[task_i].transfer_from = task_j
                # 轉換(要轉換的、要轉換過去的、轉換數量)
                ga.transfer(tasks[task_i].actor, tasks[task_j].actor, s)
                # tasks[task_i].actor[lambda_i-s :] = tasks[task_j].actor[ : s]


            ###### Evaluate ######
            tasks[task_i].evaluate_steps = 0
            print(f"Task [{args.env_names[task_i]}] evaluate:")
            print(f"Current steps: {tasks[task_i].steps}")

            for actor in tasks[task_i].actor:
                fitness, evaluate_steps = tasks[task_i].evaluate(1, actor, replay_buffers[task_i], learning_curves[task_i])
                actor.fitness = fitness
                tasks[task_i].evaluate_steps += evaluate_steps

            # 排序(轉換過來的和原本的，由大到小) 
            # indv_ranking[task_i] = np.argsort(-tasks[task_i].actor)
            fitness_arr = np.array([actor.fitness for actor in tasks[task_i].actor])
            indv_ranking[task_i] = np.argsort(np.argsort(-fitness_arr))

        # update_Symbiosis_estimation
        ratio_help_neutral_harm = [0.25, 0.5, 1.0]

        # 更新之間關係 (九種可能性)
        for task_i in range(len(tasks)):

            print("=============================================")
            print(f"Task [{args.env_names[task_i]}] is updating relations...")
            print("=============================================")

            actor_size = len(tasks[task_i].actor)
            transfer_pos = int(actor_size - adapt_transfer_size[task_i])  # 哪一個 index 開始被轉換
            help_pos = int(ratio_help_neutral_harm[0] * actor_size)
            neutral_pos = int(ratio_help_neutral_harm[1] * actor_size)
            harm_pos = int(ratio_help_neutral_harm[2] * actor_size)

            for index in range(actor_size): 
                ranking = indv_ranking[task_i][index] # 在 offspring 中的排名
                task_j = tasks[task_i].transfer_from # 從喇裡轉換來的
                if index < help_pos:
                    if index >= transfer_pos:  # 是被交換來的
                        if ranking < help_pos:
                            mutualism[task_i][task_j] += 1
                        elif ranking < neutral_pos:
                            commensalism[task_i][task_j] += 1
                        elif ranking < harm_pos:
                            parasitism[task_i][task_j] += 1
                    else:
                        mutualism[task_i][task_i] += 1
                if index < neutral_pos:
                    if index >= transfer_pos:
                        if ranking < help_pos:
                            continue
                        elif ranking < neutral_pos:
                            neutralism[task_i][task_j] += 1
                        elif ranking < harm_pos:
                            amensalism[task_i][task_j] += 1
                    else:
                        neutralism[task_i][task_i] += 1
                if index < harm_pos:
                    if index >= transfer_pos:
                        if ranking < help_pos:
                            continue
                        elif ranking < neutral_pos:
                            continue
                        elif ranking < harm_pos:
                            competition[task_i][task_j] += 1
                    else:
                        competition[task_i][task_i] += 1

            if learning_curves[task_i].steps % args.test_performance_freq == 0:
                print(f"steps={learning_curves[task_i].learning_curve_steps[-1]}  score={learning_curves[task_i].learning_curve_scores[-1]:.3f}")



    ###### 儲存結果 ######
    if args.save_result == True:
        for i in range(len(tasks)):
            learning_curves[i].save(os.path.join(args.output_folder, f"[{args.env_names[i]}]Learning Curve.json"))

    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    if (args.save_result):
        with open(os.path.join(args.output_folder, "info.txt"), mode = "w") as file:
            file.write(f"Start date: {start_date}\n")
            file.write(f"End date: {end_date}\n")
            file.close()


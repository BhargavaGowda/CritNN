import torch
from torch import nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")
torch.set_grad_enabled(False)


popSize = 50
gens = 200

pop = []
metric = []

for i in range(popSize):

    model = nn.Sequential(
    nn.Linear(8,4),
    nn.Tanh(),
    nn.Linear(4,2),
    nn.Tanh()
    )

    pop.append(model)
    metric.append(0)


    
bestFitness = -1000


for g in range(gens):
    print("progress:",str(100*g/float(gens))+"%")
    for i in range(popSize):

        model = pop[i]

        # param stepping
        # params = []
        # for name,param in model.named_parameters():
        #     params.append(param)

        observation, info = env.reset(seed=2)
        fitness = 0
        while True:

            inp = np.array(observation,dtype=np.float32)
            inp = torch.from_numpy(inp)
            output = model(inp)
            action = output.detach().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward

            if terminated or truncated:
                break

        # metric[i]= observation[15]
        metric[i] = fitness
        if fitness>bestFitness:
            bestFitness = fitness
            torch.save(model,"bestOne.modl")
    print(bestFitness)
    
    for i in range(popSize):
        bestMatch = 0
        bestCrit = 0

        p1 = np.empty(0)
        for name,param in pop[i].named_parameters():
            p1 = np.concatenate((p1,param.data.detach().clone().numpy().flatten()))

        

        for o in range(popSize):
            if i==o:
                continue

            p2 = np.empty(0)
            for name,param in pop[o].named_parameters():
                p2 = np.concatenate((p2,param.data.detach().numpy().flatten()))

            gd = np.linalg.norm(p2-p1)
            bd = np.linalg.norm(metric[o]-metric[i])
            crit = abs(bd/gd)
            
            if crit > bestCrit:
                bestCrit = crit
                bestMatch = o
        
        p2 = np.empty(0)
        for name,param in pop[bestMatch].named_parameters():
            p2 = np.concatenate((p2,param.data.detach().clone().numpy().flatten()))

        # print("a:",p2)

        delta = 0.1*(p2-p1)
        p1+=delta
        p2-=delta
        if i == 0:
            print("delta:",np.linalg.norm(delta))

        pin1 = p1.tolist()
        pin2 = p2.tolist()

        params_i = []
        for name,param in pop[i].named_parameters():
            params_i.append(param)

        params_best = []
        for name,param in pop[bestMatch].named_parameters():
            params_best.append(param)


        for pset in params_i:
            length = np.prod(pset.data.shape)
            newPset = np.zeros(length,dtype=np.float32)
            for _ in range(length):
                newPset[_] = pin1.pop(0)
            newPset = newPset.reshape(pset.data.shape)
            if i > popSize//2:
                newPset+= 10*(np.random.random(newPset.shape)- 0.5)
            pset.data = torch.from_numpy(newPset)

        for pset in params_best:
            length = np.prod(pset.data.shape)
            newPset = np.zeros(length,dtype=np.float32)
            for _ in range(length):
                newPset[_] = pin2.pop(0)
            newPset = newPset.reshape(pset.data.shape)
            pset.data = torch.from_numpy(newPset)

        # ptest = np.empty(0)
        # for name,param in pop[bestMatch].named_parameters():
        #     ptest = np.concatenate((ptest,param.data.detach().numpy().flatten()))
        # print("b:",ptest)

        
            
    

                


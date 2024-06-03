import torch
from torch import nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True)
torch.set_grad_enabled(False)

grad_x=np.zeros(30)
grad_y=np.zeros(30)

pset = 0
p1 = 0
p2 = 0
fig,axs = plt.subplots(4,2)
axs = axs.flatten()

for ax in axs:

    for i in range(30):
        print(i)
        a=0
        b=0

        model = torch.load("bestLander.modl")
        step = 1/(1.5)**i

        # param stepping
        params = []
        for name,param in model.named_parameters():
            params.append(param)

        params[pset].data[p1,p2]+= step

        observation, info = env.reset(seed=1)
        fitness = 0
        while True:

            inp = np.array(observation,dtype=np.float32)
            inp = torch.from_numpy(inp)
            output = model(inp)
            action = output.detach().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            fitness+= reward

            if terminated or truncated:
                break
        a = fitness

        params[pset].data[p1,p2]-= 2* step

        observation, info = env.reset(seed=1)
        fitness = 0
        while True:

            inp = np.array(observation,dtype=np.float32)
            inp = torch.from_numpy(inp)
            output = model(inp)
            action = output.detach().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            fitness+= reward

            if terminated or truncated:
                break
        b = fitness

        grad_x[i] = step
        grad_y[i] = (a-b)/(2*step)

    ax.plot(grad_x,grad_y)
    ax.set_xscale("log",base=10)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel("step size")
    ax.set_ylabel("gradient")
    
    p2+=1



plt.show()
            
                


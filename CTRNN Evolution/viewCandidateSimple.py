from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
# env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
# env = gym.make("BipedalWalker-v3",render_mode="human")
env = gym.make("BipedalWalker-v3",hardcore=True,render_mode="human")


inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]


# net = CTRNN(10)
# net.mutateSimple(3)
# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

with open("best_net.pkl", "rb") as f:
    net = pickle.load(f)
    net.reset()

print(net.size)
print(net.timescale)

while True:
    seed = np.random.randint(0,100000)
    # seed = 3
    print("seed:",seed)
    
    observation, info = env.reset(seed=seed)
    fitness = 0
    net.reset()

    _=0
    while True:

        inp = np.array(observation)
        # print(observation[0])
        action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= reward


        if terminated or truncated:
            break

        # print(_)
        _+=1

    print(fitness)


        




        

            
                


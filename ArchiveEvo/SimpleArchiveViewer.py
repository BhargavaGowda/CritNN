from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle

env = gym.make("Hopper-v4",render_mode="human")
# env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
# env = gym.make("BipedalWalker-v3",render_mode="human")
# env = gym.make("BipedalWalker-v3",hardcore=True,render_mode="human")
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]



with open("ArchiveEvo/CVTarchive.pkl", "rb") as f:
    archive = pickle.load(f)

# coords = (-326,-266)

# found = False
# for i in range(len(archive)):
#     if archive[i][1]==coords[0] and archive[i][2]==coords[1]:
#         found=True
#         net = archive[i][0]

# if not found:
#     print("Coords not found. Best net displayed.")
#     net = archive[0][0]

print(len(archive))
archive.sort(key=lambda x :-x[2])
neti=0
while True: 
    observation, info = env.reset()
    fitness = 0
    net = archive[neti][1]
    neti+=1
    net.reset()


    for _ in range(500):

        inp = np.array(observation)
        action = net.step(np.concatenate((inp,np.zeros(net.size-inps))))
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= reward


        if terminated or truncated:
            print(terminated)
            break

    print(fitness)


        




        

            
                


from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("LunarLander-v2",continuous=True)
env = gym.make("BipedalWalker-v3")


inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]


# net = CTRNN(10)
# net.mutateSimple(3)
# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

with open("best_fit.pkl", "rb") as f:
    net = pickle.load(f)
    net.reset()

print(net.size)

numTrials = 50
data = np.zeros(numTrials)

for t in range(numTrials):
    print("progress:",str(100*t/float(numTrials))+"%")
   
    observation, info = env.reset()
    fitness = 0
    net.reset()

    while True:

        inp = np.array(observation)
        # print(observation[0])
        action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= reward


        if terminated or truncated:
            break
    data[t] = fitness

plt.boxplot(data)
plt.show()


    


        




        

            
                


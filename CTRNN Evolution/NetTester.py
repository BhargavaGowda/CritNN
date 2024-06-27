from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("LunarLander-v2",continuous=True)
env = gym.make("HalfCheetah-v4")
# env = gym.make("InvertedDoublePendulum-v4")


inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]


numTrials = 100
data = np.zeros(numTrials)
data2 = np.zeros(numTrials)


with open("net.pkl", "rb") as f:
    net = pickle.load(f)
    net.mutateSimple(0.0001)
    net.reset()

observation, info = env.reset(seed=4)
baseFit = 0

while True:

    inp = np.array(observation)
    # print(observation[0])
    action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
    # action = np.zeros(outs)
    observation, reward, terminated, truncated, info = env.step(action[-outs:])
    baseFit+= reward


    if terminated or truncated:
        break

for t in range(numTrials):
    print("progress:",str(100*t/float(numTrials))+"%")
   
    testNet = CTRNN.recombine(net,net)
    testNet.mutateSimple(0.001)
    observation, info = env.reset(seed=4)
    fitness = 0
    

    while True:

        inp = np.array(observation)
        # print(observation[0])
        action = testNet.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= reward


        if terminated or truncated:
            break
    data[t] = abs(fitness-baseFit)/CTRNN.getDistance(net,testNet)
    data2[t] = abs(fitness-baseFit)

np.savetxt("data/Cheetah2.txt",data)
plt.plot(data)
plt.plot(data2)
plt.show()

    


        




        

            
                


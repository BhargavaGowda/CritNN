from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle
from myDoublePendulum import myPendulum

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
# env = gym.make("Ant-v4",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
# env = gym.make("BipedalWalker-v3",render_mode="human")
# env = gym.make("BipedalWalker-v3",hardcore=True,render_mode="human")
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]


netSize = 15
numSteps = 500
crossPoints = int(pow(netSize,0.6))
mutPoints =  int(netSize**0.4)

print("network of size:",netSize,"with ",crossPoints,"-point crossover and ",mutPoints,"-point mutation.")

net = CTRNN(netSize)
fitCurve = []
mutateNeuron = -1

while True: 
    observation, info = env.reset(seed=3)
    fitness = 0

    newNet = CTRNN.copy(net)
    for m in range(1):
        newNet.mutateModular(neuron=mutateNeuron)


    for _ in range(500):

        alpha=1
        inp = alpha * np.array(observation) + (1-alpha)*np.random.normal(size=observation.shape)
        action = newNet.step(np.concatenate((inp,np.zeros(netSize-inps))))
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= reward


        if terminated or truncated:
            print(terminated)
            break

    print(fitness)
    fitCurve.append(fitness)

    action = input("?:")
    if action == "y":
        net = newNet
        # nextNet = CTRNN.recombine(net,newNet)
        # for c in range(crossPoints-1):
        #     nextNet = CTRNN.recombine(nextNet,net) if np.random.rand()>0.5 else CTRNN.recombine(nextNet,newNet)
    elif(action == "q"):
        with open("best_net.pkl", "wb") as f:
            pickle.dump(net, f)
        plt.plot(fitCurve)
        plt.show()
        break
    else:
        action = int(action)
        mutateNeuron = action
    
        


        




        

            
                


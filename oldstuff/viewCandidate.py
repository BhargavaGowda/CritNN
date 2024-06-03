from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
# env = gym.make("BipedalWalker-v3",render_mode="human")

inps = env.observation_space.shape[0]+1
outs = env.action_space.shape[0]

# normVec = np.array([1/1.5,1/1.5,1/5,1/5,1/np.pi,1/5,1,1])
normVec = np.ones(env.observation_space.shape[0])


# net = CTRNN(104,weightRange=10,biasRange=10)
# net.mutateSimple(3)
# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

with open("modelArchive\\10 Neuron general with timescale mutation\\best_net.pkl", "rb") as f:
    net = pickle.load(f)
    net.reset()

print(net.size)

while True:
    seed = np.random.randint(0,10000)
    # seed = 70862
    # print("seed:",seed)
    
    perturb = 1.5*(np.random.rand()-0.5)
    print("--")
    print("control:",perturb)
    observation, info = env.reset(seed=seed)
    fitness = 0
    net.reset()

    _=0
    while True:
        net.inputs[8] = perturb

        inp = np.array(observation)*normVec
        # print(observation[0])
        action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= reward

        

        

        if terminated or truncated:
            break 

        if terminated or truncated:
            print(fitness)
            break

        _+=1

    print(observation[0])


        




        

            
                


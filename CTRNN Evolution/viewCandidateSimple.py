from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle
from myDoublePendulum import myPendulum

# env = gym.make("HalfCheetah-v4",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
# env = gym.make("BipedalWalker-v3",render_mode="human")
# env = gym.make("BipedalWalker-v3",hardcore=True,render_mode="human")
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

while True: 
    observation, info = env.reset()
    fitness = 0
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


        




        

            
                


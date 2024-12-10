from PyANN import ANN
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
# env = gym.make("Walker2d-v4",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
# env = gym.make("BipedalWalker-v3",render_mode="human")
# env = gym.make("BipedalWalker-v3",hardcore=True,render_mode="human")
# env = gym.make("Ant-v4",exclude_current_positions_from_observation=False,render_mode="human")
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]


with open("ArbitFit/best_net.pkl", "rb") as f:
    net = pickle.load(f)

with open("ArbitFit/fit_net.pkl", "rb") as f:
    fitnet = pickle.load(f)


print(fitnet.size,net.size)

while True: 
    observation, info = env.reset()
    fitness = 0
    # net.mutateSimple(0.1)



    for _ in range(500):

        inp = np.array(observation)
        action = net.step(inp)
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= fitnet.step(inp)

        if terminated or truncated:
            break


    
    print(fitness)


        




        

            
                


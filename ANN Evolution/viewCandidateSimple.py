from PyANN import ANN
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
env = gym.make("Ant-v4",render_mode="human")
# env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
# env = gym.make("BipedalWalker-v3",render_mode="human")
# env = gym.make("BipedalWalker-v3",hardcore=True,render_mode="human")
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]



with open("best_fit.pkl", "rb") as f:
    net = pickle.load(f)

# net = ANN([inps,outs])
# net.mutateSimple()

print(net.size)

while True: 
    observation, info = env.reset()
    fitness = 0


    for _ in range(500):

        alpha=1
        inp = alpha * np.array(observation) + (1-alpha)*np.random.normal(size=observation.shape)
        action = net.step(inp)
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        fitness+= reward


        if terminated or truncated:
            print(terminated)
            break
    
    print(fitness)


        




        

            
                


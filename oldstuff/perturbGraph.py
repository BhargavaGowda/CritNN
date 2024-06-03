from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0")
env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")


inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]

normVec = np.array([1/1.5,1/1.5,1/5,1/5,1/np.pi,1/5,1,1])
# normVec = np.ones(env.observation_space.shape[0])

# net = CTRNN(30)
# net.mutateSimple(3)
# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

perturbs = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
results = []

with open("modelArchive/best_net1.pkl", "rb") as f:
    net = pickle.load(f)
    net.reset()

print(net.size)





numTrials = 50


    
for i in range(len(perturbs)):
    print("progress:",str(100*i/float(len(perturbs)))+"%")

    Sum = 0
    for j in range(numTrials):
        seed = np.random.randint(0,10000)
        # first run
        observation, info = env.reset(seed=seed)
        net.reset()
        fitness =0

        for _ in range(200):

            inp = np.array(observation)*normVec
            action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
            observation, reward, terminated, truncated, info = env.step(action[-outs:])
            fitness+=reward

            if terminated or truncated:
                break

        control = observation[0]

        #2nd run
        net.reset()
        observation, info = env.reset(seed=seed)
        fitness = 0
        for _ in range(200):

            inp = np.array(observation)*normVec
            action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
            observation, reward, terminated, truncated, info = env.step(action[-outs:])
            fitness+= reward

            if _==5:

                net.potentials[0]+=perturbs[i]

            if terminated or truncated:
                break
        Sum += abs(observation[0]-control)
    
    avg = Sum/numTrials
    results.append(avg)


        
plt.loglog(perturbs,results)
plt.show()
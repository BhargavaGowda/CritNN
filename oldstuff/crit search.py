from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True)
normVec = np.array([1/1.5,1/1.5,1/5,1/5,1/np.pi,1/5,1,1])
# env = gym.make("BipedalWalker-v3")


popSize = 50
gens = 100
netSize = 200
perturbSize = 0.001

pop = []
metric = []
crit = []

for i in range(popSize):

    net = CTRNN(netSize)
    net.mutate(1)
    net.mutateSimple(5)
    pop.append(net)
    metric.append(0)
    crit.append(0)


    
bestFitness = -1000
bestCrit = 0


for g in range(gens):
    print("progress:",str(100*g/float(gens))+"%")


    for i in range(popSize):

        net = pop[i]

        # first run
        observation, info = env.reset(seed=3)
        net.reset()
        fitness = 0
        while True:

            inp = np.array(observation)
            action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
            observation, reward, terminated, truncated, info = env.step(action[-4:])
            fitness+= reward

            if terminated or truncated:
                break

        metric[i] = observation

        if fitness>bestFitness:
            bestFitness = fitness
            with open("bestNet.pkl", "wb") as f:
                pickle.dump(net, f)

        #2nd run
        observation, info = env.reset(seed=3)
        net.reset()
        fitness = 0
        _ = 0
        while True:

            inp = np.array(observation)
            action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
            observation, reward, terminated, truncated, info = env.step(action[-4:])
            fitness+= reward

            if _==5:

                net.potentials[np.random.randint(0,netSize)]+=perturbSize

            if terminated or truncated:
                break
            _+=1

        crit[i] = np.linalg.norm(metric[i] - observation)

        if crit[i]>bestCrit:
            bestCrit = crit[i]
            with open("critestNet.pkl", "wb") as f:
                pickle.dump(net, f)
             




    for i in range(popSize):
        match = None

        for o in range(popSize):
            if i != o:
                if crit[o]>crit[i]:
                    match = pop[o]

        if match:
            # newNet = CTRNN.recombine(pop[i],match)
            # newNet.mutateSplit()
            # if i<10:
            #      newNet.mutateSplit(5,0.2)
            # pop[i] = newNet

            pop[i].lerpNet(match,0.1)
            pop[i].mutateSplit()
            if i<10:
                 pop[i].mutateSplit(3,0.2)

        

    print("best fit:",bestFitness)
    print("best crit:",bestCrit)




        
            
    

                


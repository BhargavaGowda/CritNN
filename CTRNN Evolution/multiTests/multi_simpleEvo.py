from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("LunarLander-v2",continuous=True)
env = gym.make("BipedalWalker-v3")



popSize = 50
gens = 200
netSize = 30
numMutPoints = int(pow(netSize,0.6))
print("numMutPoints:",numMutPoints)
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]
trials = 5
data = np.zeros(trials)

for trial in range(trials):
    print("progress:",str(100*trial/float(trials))+"%")

    pop = []

    for i in range(popSize):


        net = CTRNN(netSize)
        net.mutateSimple(1)
        net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
        net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
        pop.append([net,0])



        
    bestFitness = -1000


    for g in range(gens):
        


        for i in range(popSize):
            seed = np.random.randint(0,100000)

            net = pop[i][0]

            # first run
            observation, info = env.reset(seed=seed)
            net.reset()
            fitness = 0
            
            # while True:
            for _ in range(300):

                inp = np.array(observation)
                action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
                observation, reward, terminated, truncated, info = env.step(action[-outs:])
                fitness+= reward

                if terminated or truncated:
                    break
            
            if fitness>bestFitness:
                bestFitness=fitness
            pop[i][1] = fitness

        pop.sort(key= lambda x : -x[1])

        newPop = []
        newPop.append([pop[0][0],0])

        for i in range(popSize-1):

            a = np.random.randint(0,popSize//2)
            b = np.random.randint(0,popSize//2)
            while a == b:
                b= np.random.randint(0,popSize//2)

            newNet = CTRNN.recombine(pop[a][0],pop[b][0])

            mutRate = 1

            if i >popSize//2:
                for i in range(numMutPoints):
                    newNet.mutatePointFull(mutRate)    

            newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
            newNet.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))

            newPop.append([newNet,0])

        pop = newPop
    
    data[trial] = bestFitness

np.savetxt("simpleEvoTrialsResults.txt",data)
    





        
            
    

                


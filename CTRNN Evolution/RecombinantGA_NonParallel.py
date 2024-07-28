from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")
env = gym.make("HalfCheetah-v4")

popSize = 50
gens = 200


netSize = 25
crossPoints = int(pow(netSize,0.6))
mutPoints =  int(netSize**0.5)

print("network of size:",netSize,"with ",crossPoints,"-point crossover and ",mutPoints,"-point mutation.")

diversityThreshold = 0.1*(netSize**2)
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]
pop = []
data = np.zeros((gens,2))

for i in range(popSize):

    net = CTRNN(netSize)
    net.mutateSimple(5)
    net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
    net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
    pop.append([net,None])


    
bestFitness = -1000


for g in range(gens):
    print("progress:",str(100*g/float(gens))+"%")

    for i in range(popSize):

        net = pop[i][0]

        # first run
        observation, info = env.reset()
        net.reset()
        fitness = 0
        while True:

            inp = np.array(observation)
            action = net.step(np.concatenate((inp,np.zeros(net.size-inps))))
            # action = np.zeros(outs)
            observation, reward, terminated, truncated, info = env.step(action[-outs:])
            fitness+= reward


            if terminated or truncated:
                break

        pop[i][1] = fitness
        if fitness>bestFitness:
            bestFitness = fitness
            with open("best_fit.pkl", "wb") as f:
                pickle.dump(net, f)
    
    pop.sort(key= lambda x : -x[1])
    data[g,0] = bestFitness
    print("bestFit:",bestFitness)
    avgDiv = np.mean([CTRNN.getGeneDistance(pop[i][0],pop[i+1][0]) for i in range(popSize-1)])
    print("Diversity:",avgDiv)
    data[g,1]=avgDiv


    newPop = []
    newPop.append([pop[0][0],None])

    for i in range(popSize-1):

        a = np.random.randint(0,popSize//2)
        b = np.random.randint(0,popSize//2)

        
        while a == b:
            b= np.random.randint(0,popSize//2)

        # newNet = CTRNN.copy(pop[a][0])
        # newNet.mutateSimple()

        newNet = CTRNN.recombine(pop[a][0],pop[b][0])
        for c in range(crossPoints):
            newNet = CTRNN.recombine(newNet,pop[a][0]) if np.random.rand()>0.5 else CTRNN.recombine(newNet,pop[b][0])

        if avgDiv<diversityThreshold:
                newNet.mutateModular()

        newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
        newNet.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
        newPop.append([newNet,None])

    pop = newPop

        

print("best fit individual found:",bestFitness)
with open("best_net.pkl", "wb") as f:
    pickle.dump(pop[0][0], f)

np.savetxt("data/RecombinantGA_Results.txt",data)
plt.plot(data[:,0])
plt.plot(data[:,1])
plt.show()





        
            
    

                


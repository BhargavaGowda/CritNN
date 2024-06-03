from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# env = gym.make("MountainCarContinuous-v0")
env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")

popSize = 100
gens = 10000
fitCurve = np.zeros(gens)
netSize = 30
numPoints = 1
numMutPoints = int(pow(netSize,0.6))
print("numMutPoints:",numMutPoints)
pop = []

inps = env.observation_space.shape[0]+1
outs = env.action_space.shape[0]

normVec = np.array([1/1.5,1/1.5,1/5,1/5,1/np.pi,1/5,1,1])
# normVec = np.ones(inps)




for i in range(popSize):

    net = CTRNN(netSize)
    net.mutateSplit(5,0)
    net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
    net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
    pop.append([net,-500])




    
bestFitness = -1000


for g in range(gens):
    print("progress:",str(100*g/float(gens))+"%")


    for i in range(popSize):

        net = pop[i][0]
        
        errors = 0
        for n in range(numPoints):
            seed = np.random.randint(0,10000)
            control = 1.5*(np.random.rand()-0.5)
            net.reset()
            observation, info = env.reset(seed=seed)
            fitness = 0
            while True:
                
                net.inputs[8]=control
                inp = np.array(observation)*normVec
                action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
                observation, reward, terminated, truncated, info = env.step(action[-outs:])
                fitness+= reward
                
                if terminated or truncated:
                    break
        
            errors += abs(control-observation[0])

        pop[i][1] = errors/numPoints

    pop.sort(key= lambda x : x[1])
    fitCurve[g] = sum(i[1] for i in pop)/len(pop)
    # fitCurve[g] = pop[0][1]
    print(fitCurve[g])

    

    newPop = []
    newPop.append([pop[0][0],0])

    for i in range(popSize-1):

        a = np.random.randint(0,popSize//2)
        b = np.random.randint(0,popSize//2)
        while a == b:
            b= np.random.randint(0,popSize//2)

        newNet = CTRNN.recombine(pop[a][0],pop[b][0])

        mutRate = 1#1-(0.9*g)/gens

        if i >popSize//2:
            for i in range(numMutPoints):
                newNet.mutatePointFull(mutRate)    

        newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
        newNet.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))

        newPop.append([newNet,0])

    pop = newPop
    # for i in range(10):
    #     print(pop[i][0].weights[9])
    
with open("best_net.pkl", "wb") as f:
    pickle.dump(pop[0][0], f)

    
plt.plot(fitCurve)
plt.xlabel("generation")
plt.ylabel("avg error of pop")
# plt.title("Seed:" + str(seed))
plt.title("random seed")
plt.savefig('fitcurveLinearEvo.png', bbox_inches='tight')
plt.show()






        
            
    

                


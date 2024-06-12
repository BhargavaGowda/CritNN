from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("InvertedPendulum-v4")
# env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")

inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]

popSize = 100
gens = 300
netSize = 30
fitCurve = np.zeros(gens)
bestFitCurve = np.zeros(gens)

net = CTRNN(netSize)
net.mutateSimple(3)
net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))

pop = []


bestFitness = -10000

for i in range(popSize):

    init = np.random.normal(size=net.potentials.shape)
    pop.append([init,0])



# seed = np.random.randint(0,100000)
# seed = 60759
    
bestFitness = -1000


for g in range(gens):
    print("progress:",str(100*g/float(gens))+"%")


    for i in range(popSize):
        seed = np.random.randint(0,100000)

        # first run
        observation, info = env.reset(seed=seed)
        net.reset()
        # ___
        net.potentials = pop[i][0]
        observation, reward, terminated, truncated, info = env.step(net.potentials[-outs:])

        fitness = 0
        
        while True:
        # for _ in range(300):

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
    bestFitCurve[g] = bestFitness
    fitCurve[g] = sum(pop[i][1] for i in range(popSize//2))/float(popSize//2)
    print(fitCurve[g],bestFitCurve[g])

    newPop = []
    newPop.append([pop[0][0],0])

    for i in range(popSize-1):

        a = np.random.randint(0,popSize//2)
        newInit = pop[a][0]

        mutRate = 1#1-(0.9*g)/gens

        if i >popSize//2:
            newInit+=np.random.normal(size=net.potentials.shape)*mutRate

        newPop.append([newInit,0])

    pop = newPop
    # for i in range(10):
    #     print(pop[i][0].weights[9])
    
with open("best_init.pkl", "wb") as f:
    pickle.dump(pop[0][0], f)

    
plt.plot(fitCurve)
plt.plot(bestFitCurve)
plt.xlabel("generation")
plt.ylabel("fitness")
plt.legend(["population mean","fittest net"])
# plt.title("Seed:" + str(seed))
# plt.title("random seed")
plt.savefig('fitcurveSimpleEvo.png', bbox_inches='tight')
plt.show()





        
            
    

                


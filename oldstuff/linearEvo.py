from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# env = gym.make("MountainCarContinuous-v0")
env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")


zeroBias = False
# env = gym.make("BipedalWalker-v3")


popSize = 200
gens = 10000
fitCurve = np.zeros(gens)
netSize = 100
pop = []

inps = env.observation_space.shape[0]+1
outs = env.action_space.shape[0]

normVec = np.array([1/1.5,1/1.5,1/5,1/5,1/np.pi,1/5,1,1])
# normVec = np.ones(inps)




for i in range(popSize):

    net = CTRNN(netSize)
    net.mutateSplit(5,0)
    if zeroBias:
        net.bias = np.zeros(net.size)
    net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
    net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
    pop.append([net,-500])




    
bestFitness = -1000


for g in range(gens):
    print("progress:",str(100*g/float(gens))+"%")


    for i in range(popSize):

        net = pop[i][0]
        control = 2*(np.random.rand()-0.5)

        seed = 3#np.random.randint(0,10000)

        # # first run
        # observation, info = env.reset(seed=seed)
        # net.reset()
        # fitness =0

        # for _ in range(200):

        #     inp = np.array(observation)*normVec
        #     action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
        #     observation, reward, terminated, truncated, info = env.step(action[-outs:])
        #     fitness+=reward

        #     if terminated or truncated:
        #         break

        # control = observation[0]

        #2nd run
        net.reset()
        observation, info = env.reset(seed=seed)
        fitness = 0
        while True:

            inp = np.array(observation)*normVec
            action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
            observation, reward, terminated, truncated, info = env.step(action[-outs:])
            fitness+= reward

            net.inputs[8]=control

            if terminated or truncated:
                break
        
        pop[i][1] = abs(control-observation[0])

    pop.sort(key= lambda x : x[1])
    fitCurve[g] = sum(i[1] for i in pop)/len(pop)
    print(fitCurve[g])

    

    newPop = []
    newPop.append([pop[0][0],0])


    for i in range(popSize-1):

        a = np.random.randint(0,popSize//2)
        b = np.random.randint(0,popSize//2)
        while a == b:
            b= np.random.randint(0,popSize//2)

        newNet = CTRNN.recombine(pop[a][0],pop[b][0])

        if i >popSize//2:
            if g > gens//2:
                newNet.mutateSplit(0.5,0)
            else:
                newNet.mutateSplit(1,0)
                
            if zeroBias:
                newNet.bias = np.zeros(net.size)

        net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
        net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))

        newPop.append([newNet,-500])

    pop = newPop
    
with open("best_net1.pkl", "wb") as f:
    pickle.dump(pop[0][0], f)

with open("best_net2.pkl", "wb") as f:
    pickle.dump(pop[1][0], f)

with open("best_net3.pkl", "wb") as f:
    pickle.dump(pop[2][0], f)

with open("best_net4.pkl", "wb") as f:
    pickle.dump(pop[3][0], f)

with open("best_net5.pkl", "wb") as f:
    pickle.dump(pop[4][0], f)

plt.plot(fitCurve)
plt.savefig('fitcurveLinearEvo.png', bbox_inches='tight')
plt.show()






        
            
    

                


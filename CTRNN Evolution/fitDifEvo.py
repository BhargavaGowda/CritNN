from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")
env = gym.make("BipedalWalker-v3",hardcore=True)


zeroBias = False

popSize = 50
gens = 100
netSize = 30
fitCurve = np.zeros(gens)
fitDifCurve = np.zeros(gens)
numMutPoints = int(pow(netSize,0.6))
print("numMutPoints:",numMutPoints)
pop = []

inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]




for i in range(popSize):

    net = CTRNN(netSize)
    net.mutateSimple(1)
    if zeroBias:
        net.bias = np.zeros(net.size)
    net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
    net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
    pop.append([net,0,0])



# seed = np.random.randint(0,100000)
# seed = 60759
    
bestFitness = -1000


for g in range(gens):
    print("progress:",str(100*g/float(gens))+"%")


    for i in range(popSize):
        seed = np.random.randint(0,100000)

        net = pop[i][0]
        net2=CTRNN.recombine(net,net)
        net2.mutatePointFull(1)

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
            if fitness<-200:
                break

        firstFit = fitness

        # 2nd run
        observation, info = env.reset(seed=seed)
        net2.reset()
        fitness = 0
        
        # while True:
        for _ in range(300):

            inp = np.array(observation)
            action = net2.step(np.concatenate((inp,np.zeros(net2.size-inp.size))))
            observation, reward, terminated, truncated, info = env.step(action[-outs:])
            fitness+= reward

            if terminated or truncated:
                break
            if fitness<-200:
                break

        pop[i][1] = abs(firstFit-fitness)/CTRNN.getDistance(net,net2)
        pop[i][2] = firstFit

        if pop[i][2]>bestFitness:
            with open("best_fit.pkl", "wb") as f:
                pickle.dump(net, f)
            bestFitness=pop[i][2]
        


    fitCurve[g] = bestFitness#sum(pop[i][2] for i in range(popSize//2))/float(popSize//2)
    print(fitCurve[g])
    
    
    pop.sort(key= lambda x : -x[1])
    newPop = []
    newPop.append([pop[0][0],0,0])

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

                   
        if zeroBias:
            newNet.bias = np.zeros(net.size)

        newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
        newNet.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))

        newPop.append([newNet,0,0])

    pop = newPop
    # for i in range(10):
    #     print(pop[i][0].weights[9])
    
with open("best_net.pkl", "wb") as f:
    pickle.dump(pop[0][0], f)

    
plt.plot(fitCurve)
# plt.plot(fitDifCurve)
plt.xlabel("generation")
plt.ylabel("avg fitness of pop")
# plt.title("Seed:" + str(seed))
plt.title("random seed")
plt.savefig('fitcurveDifEvo.png', bbox_inches='tight')
plt.show()





        
            
    

                


from PyCTRNNv3 import CTRNN
import pickle
import math
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#14 S/s

def main():

    #Search Parameters
    popSize = 50
    gens = 500
    netSize = 10
    numSteps = 500

    numMutPoints = int(pow(netSize,0.6))
    print("numMutPoints:",numMutPoints)
    crossover = False
    importNet = False

    numRuns = 30
    data = np.zeros((numRuns,gens))

    for run in range(numRuns):
        print("Run:",run)


        bestFitness = -10000
        bestFitCurve = np.zeros(gens)
        # envs = gym.vector.make("Ant-v4",num_envs = popSize)
        envs = gym.vector.make("LunarLander-v2",continuous=True,num_envs=popSize)
        # envs = gym.vector.make("BipedalWalker-v3",num_envs = popSize)

        # Initializing Population
        inps = envs.observation_space.shape[1]
        outs = envs.action_space.shape[1]

        pop = []
        for i in range(popSize):

            if importNet:
                with open("best_fit.pkl", "rb") as f:
                    net = pickle.load(f)
                    net.reset()
            else:
                net = CTRNN(netSize)
                net.mutateSimple(1)
            net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
            net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
            pop.append([net,0,0])


        #Running gens
        for g in range(gens):
            print("progress:",str(100*g/float(gens))+"%")

            #Main Evaluation

            observations, infos = envs.reset()
            fits = np.zeros(popSize)
            dones = np.ones(popSize)
            for i in range(popSize):
                pop[i][0].reset()

            for _ in range(numSteps):

                actions = []
                for i in range(popSize):
                    action = pop[i][0].step(np.concatenate((observations[i],np.zeros(net.size-inps))))[-outs:]
                    actions.append(action)

                observations, rewards, terminateds, truncateds, infos = envs.step(actions)
                fits += dones*rewards
                fits = fits.clip(min=-100)

                for d in range(popSize):
                    if terminateds[d] or truncateds[d]:
                        dones[d]=0

            for p in range(popSize):
                pop[p][1] = fits[p]

            pop.sort(key= lambda x : -x[1])
            
            #Checking best fit
            if pop[0][1]>bestFitness:

                testPop = []
                for i in range(popSize):
                    testPop.append(CTRNN.recombine(pop[0][0],pop[0][0]))

                observations, infos = envs.reset()
                fits = np.zeros(popSize)
                dones = np.ones(popSize)

                for _ in range(numSteps):

                    actions = []
                    for i in range(popSize):
                        action = testPop[i].step(np.concatenate((observations[i],np.zeros(net.size-inps))))[-outs:]
                        actions.append(action)

                    observations, rewards, terminateds, truncateds, infos = envs.step(actions)
                    fits += dones*rewards

                    for d in range(popSize):
                        if terminateds[d] or truncateds[d]:
                            dones[d]=0

                if np.mean(fits)>bestFitness: 
                    with open("best_fit.pkl", "wb") as f:
                        pickle.dump(testPop[0], f)
                    bestFitness=np.mean(fits)
           
            bestFitCurve[g] = bestFitness

            for p in range(popSize):
                crit = 0
                for b in range(popSize):
                    if p!=b:
                        if math.isclose(CTRNN.getDistance(pop[p][0],pop[b][0]), 0, abs_tol=0.00003):
                            crit+=0
                        else:
                            crit+=abs(pop[p][1]-pop[b][1]/CTRNN.getDistance(pop[p][0],pop[b][0]))

                
                crit/=float(popSize-1)
                pop[p][2] = crit
            
            pop.sort(key= lambda x : -x[2])
            print("fit:",bestFitness,"crit:",pop[0][2])


            # Building Pop for next gen
            newPop = []
            newPop.append([pop[0][0],0,0])

            for i in range(popSize-1):

                a = np.random.randint(0,popSize//2)
                b = np.random.randint(0,popSize//2)

                if crossover:
                    while a == b:
                        b= np.random.randint(0,popSize//2)
                else:
                    b = a

                newNet = CTRNN.recombine(pop[a][0],pop[b][0])



                mutRate =2
                for m in range(numMutPoints):
                    newNet.mutatePointFull(mutRate)

                newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
                newNet.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))

                newPop.append([newNet,0,0])

            pop = newPop

        #End Run  
        with open("best_net.pkl", "wb") as f:
            pickle.dump(pop[0][0], f)
        data[run,:] = bestFitCurve
        print(bestFitCurve[-1])
        envs.close()

    np.savetxt("data/CritEvoResults.txt",data)


if __name__ == "__main__":
    main()





    





        
            
    

                


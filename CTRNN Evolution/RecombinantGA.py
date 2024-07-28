from PyCTRNNv3 import CTRNN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#14 S/s

def main():

    #Search Parameters
    popSize = 50
    gens = 200
    netSize = 20
    numSteps = 500
    crossPoints = int(pow(netSize,0.6))
    mutPoints =  int(netSize**0.4)

    print("network of size:",netSize,"with ",crossPoints,"-point crossover and ",mutPoints,"-point mutation.")

    diversityThreshold = 5 #0.1*(netSize**2)

    numRuns = 1
    data = np.zeros((numRuns,gens))

    for run in range(numRuns):
        print("Run:",run)


        bestFitness = -10000
        bestFitCurve = np.zeros(gens)
        envs = gym.vector.make("Hopper-v4",num_envs = popSize)
        # envs = gym.make_vec("LunarLander-v2",continuous=True,num_envs=popSize)
        # envs = gym.make_vec("BipedalWalker-v3",num_envs=popSize)

        # Initializing Population
        inps = envs.observation_space.shape[1]
        outs = envs.action_space.shape[1]

        pop = []
        for i in range(popSize):

            net = CTRNN(netSize)
            net.mutateSimple(1)
            net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
            net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
            pop.append([net,0])


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

                for d in range(popSize):
                    if terminateds[d] or truncateds[d]:
                        dones[d]=0

            for p in range(popSize):
                pop[p][1] = fits[p]

            pop.sort(key= lambda x : -x[1])

            if pop[0][1]>bestFitness: 
                with open("best_fit.pkl", "wb") as f:
                    pickle.dump(pop[0][0], f)
                bestFitness=pop[0][1]

            #Checking best fit
            # if pop[0][1]>bestFitness:

            #     testPop = []
            #     for i in range(popSize):
            #         testPop.append(CTRNN.recombine(pop[0][0],pop[0][0]))

            #     observations, infos = envs.reset()
            #     fits = np.zeros(popSize)
            #     dones = np.ones(popSize)

            #     for _ in range(numSteps):

            #         actions = []
            #         for i in range(popSize):
            #             action = testPop[i].step(np.concatenate((observations[i],np.zeros(net.size-inps))))[-outs:]
            #             actions.append(action)

            #         observations, rewards, terminateds, truncateds, infos = envs.step(actions)
            #         fits += dones*rewards

            #         for d in range(popSize):
            #             if terminateds[d] or truncateds[d]:
            #                 dones[d]=0

            #     if np.mean(fits)>bestFitness: 
            #         with open("best_fit.pkl", "wb") as f:
            #             pickle.dump(testPop[0], f)
            #         bestFitness=np.mean(fits)
           
            bestFitCurve[g] = bestFitness
            print("best fit:",bestFitness)
            avgDiv = np.mean([CTRNN.getGeneDistance(pop[i][0],pop[i+1][0]) for i in range(popSize//2-1)])
            print("Diversity:",avgDiv)


            # Building Pop for next gen
            newPop = []
            newPop.append([pop[0][0],0])

            for i in range(popSize-1):

                a = np.random.randint(0,popSize//2)
                b = np.random.randint(0,popSize//2)

                
                while a == b:
                    b= np.random.randint(0,popSize//2)

                newNet = CTRNN.recombine(pop[a][0],pop[b][0])
                for c in range(crossPoints):
                    newNet = CTRNN.recombine(newNet,pop[a][0]) if np.random.rand()>0.5 else CTRNN.recombine(newNet,pop[b][0])

                if avgDiv<diversityThreshold:
                        newNet.mutateModular()

                newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
                newNet.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
                newPop.append([newNet,None])

            pop = newPop

        #End Run  
        with open("best_net.pkl", "wb") as f:
            pickle.dump(pop[0][0], f)
        data[run,:] = bestFitCurve
        print(bestFitCurve[-1])
        envs.close()

    np.savetxt("data/SimpleEvoResults.txt",data)


if __name__ == "__main__":
    main()





    





        
            
    

                


from PyANN import ANN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#14 S/s

def main():

    #Search Parameters
    popSize = 50
    gens = 50
    netSize = [18,16,16,6]
    numSteps = 500

    numMutPoints = int(pow(sum(netSize),0.6))
    print("numMutPoints:",numMutPoints)
    crossover = False
    importNet = False
    
    
    numRuns = 1
    data = np.zeros((numRuns,gens))

    for run in range(numRuns):
        print("Run:",run)


        bestFitness = -10000
        bestFitCurve = np.zeros(gens)
        # envs = gym.vector.make("Hopper-v4",num_envs = popSize)
        # envs = gym.vector.make("LunarLander-v2",continuous=True,num_envs=popSize)
        envs = gym.vector.make("HalfCheetah-v4",exclude_current_positions_from_observation=False,num_envs = popSize)
        # envs = gym.make_vec("BipedalWalker-v3",num_envs=popSize)

        # Initializing Population
        inps = envs.observation_space.shape[1]
        outs = envs.action_space.shape[1]
        with open("ArbitFit/fit_net.pkl", "rb") as f:
            fitnet = pickle.load(f)
        print(fitnet.size)
        pop = []
        for i in range(popSize):
            net = ANN(netSize)
            net.mutateSimple(1)
            pop.append([net,0])


        #Running gens
        for g in range(gens):
            print("progress:",str(100*g/float(gens))+"%")

            #Main Evaluation

            observations, infos = envs.reset()
            fits = np.zeros(popSize)
            
            for _ in range(numSteps):

                actions = []
                for i in range(popSize):
                    action = pop[i][0].step(observations[i])
                    actions.append(action)
                    fits[i]+=fitnet.step(observations[i])

                observations, rewards, terminateds, truncateds, infos = envs.step(actions)


            for p in range(popSize):
                pop[p][1] = fits[p]

            pop.sort(key= lambda x : -x[1])
            
            if pop[0][1]>bestFitness:
                with open("ArbitFit/bestest_fit.pkl", "wb") as f:
                    pickle.dump(pop[0][0], f)
                bestFitness=pop[0][1]

            bestFitCurve[g] = bestFitness
            print("best fit:",bestFitness)


            # Building Pop for next gen
            newPop = []
            newPop.append([pop[0][0],0])

            for i in range(popSize-1):

                a = np.random.randint(0,popSize//2)
                b = np.random.randint(0,popSize//2)

                if crossover:
                    while a == b:
                        b= np.random.randint(0,popSize//2)
                    newNet = ANN.recombine(pop[a][0],pop[b][0])
                else:
                    newNet = ANN.copy(pop[a][0])

                mutRate =1
                # newNet.mutateSimple()
                for m in range(numMutPoints):
                    newNet.mutateModular(mutRate)


                newPop.append([newNet,0])

            pop = newPop

        #End Run  
        with open("ArbitFit/bestest_net.pkl", "wb") as f:
            pickle.dump(pop[0][0], f)
        data[run,:] = bestFitCurve
        print(bestFitCurve[-1])
        envs.close()


if __name__ == "__main__":
    main()





    





        
            
    

                


from PyANN import ANN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#14 S/s

def main():

    #Search Parameters
    popSize = 50
    gens = 1000
    netSize = [24,10,4]
    numSteps = 500
    importNet = False
    stepSize = 0.1
    numRuns = 1
    data = np.zeros((numRuns,gens))

    for run in range(numRuns):
        print("Run:",run)


        bestFitness = -10000
        bestFitCurve = np.zeros(gens)
        # envs = gym.vector.make("Ant-v4",num_envs = popSize)
        # envs = gym.vector.make("LunarLander-v2",continuous=True,num_envs=popSize)
        envs = gym.make_vec("BipedalWalker-v3",num_envs=popSize)

        # Initializing Population
        inps = envs.observation_space.shape[1]
        outs = envs.action_space.shape[1]

        pop = []
        mainNet = ANN(netSize)
        if importNet:
            with open("best_fit.pkl", "rb") as f:
                net = pickle.load(f)
                net.reset()
        for i in range(popSize):
            net = ANN.copy(mainNet)
            net.mutateSimple()
            pop.append([net,0])


        #Running gens
        for g in range(gens):
            print("progress:",str(100*g/float(gens))+"%")

            #Main Evaluation

            observations, infos = envs.reset()
            fits = np.zeros(popSize)
            dones = np.ones(popSize)

            for _ in range(numSteps):

                actions = []
                for i in range(popSize):
                    action = pop[i][0].step((observations[i]))
                    actions.append(action)

                observations, rewards, terminateds, truncateds, infos = envs.step(actions)
                fits += dones*rewards

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
                    testPop.append(ANN.copy(pop[0][0]))

                observations, infos = envs.reset()
                fits = np.zeros(popSize)
                dones = np.ones(popSize)

                for _ in range(numSteps):

                    actions = []
                    for i in range(popSize):
                        action = pop[i][0].step((observations[i]))
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
            print("best fit:",bestFitness)


            # Updating center
            fits = fits/np.linalg.norm(fits)
            for p in range(popSize):
                for i in range(len(mainNet.size)-1):
                    mainNet.weights[i]+=stepSize*(pop[p][0].weights[i]-mainNet.weights[i])*fits[p]
                    mainNet.bias[i]+=stepSize*(pop[p][0].bias[i]-mainNet.bias[i])*fits[p]

            for i in range(popSize):
                net = ANN.copy(mainNet)
                net.mutateSimple()
                pop[i] = [net,0]


        #End Run  
        with open("main_net.pkl", "wb") as f:
            pickle.dump(mainNet, f)
        data[run,:] = bestFitCurve
        print(bestFitCurve[-1])
        envs.close()

    np.savetxt("data/ES_Results.txt",data)


if __name__ == "__main__":
    main()





    





        
            
    

                


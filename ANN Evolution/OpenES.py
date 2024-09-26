import numpy as np
from es import OpenES
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
from PyANN import ANN


def main():
    


    netSize = [27,64,64,8]
    numRuns = 1
    gens = 10000
    data = np.zeros((numRuns,gens))
    testNet = ANN(netSize)
    paramlength = testNet.getParamVec().size
    popSize = 50
    numSteps = 500

    for run in range(numRuns):
        es = OpenES(paramlength,                  # number of model parameters
                sigma_init=1.,            # initial standard deviation
                sigma_decay=0.995,         # don't anneal standard deviation
                learning_rate=0.1,         # learning rate for standard deviation
                learning_rate_decay = 1.0, # annealing the learning rate
                popsize=popSize,       # population size
                antithetic=False,          # whether to use antithetic sampling
                weight_decay=0.00,         # weight decay coefficient
                rank_fitness=False,        # use rank rather than fitness numbers
                forget_best=False)
        
        print("Run:",run)


        bestFitCurve = np.zeros(gens)

        envs = gym.vector.make("Ant-v4",num_envs = popSize)
        # envs = gym.vector.make("LunarLander-v2",continuous=True,num_envs=popSize)
        # envs = gym.make_vec("BipedalWalker-v3",num_envs=popSize)

        popVecs = es.ask()
        pop = []        
        for i in range(popSize):
            net = ANN(netSize)
            net.setParamVec(popVecs[i])
            pop.append(net)



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
                    action = pop[i].step((observations[i]))
                    actions.append(action)

                observations, rewards, terminateds, truncateds, infos = envs.step(actions)
                fits += dones*rewards

                for d in range(popSize):
                    if terminateds[d] or truncateds[d]:
                        dones[d]=0

            es.tell(fits)
            bestFitCurve[g]= np.copy(es.result()[1])
            print(bestFitCurve[g])
            
            #Checking best fit
            # if pop[0][1]>bestFitness:

            #     testPop = []
            #     for i in range(popSize):
            #         testPop.append(ANN.copy(pop[0][0]))

            #     observations, infos = envs.reset()
            #     tfits = np.zeros(popSize)
            #     dones = np.ones(popSize)

            #     for _ in range(numSteps):

            #         actions = []
            #         for i in range(popSize):
            #             action = pop[i][0].step((observations[i]))
            #             actions.append(action)

            #         observations, rewards, terminateds, truncateds, infos = envs.step(actions)
            #         tfits += dones*rewards

            #         for d in range(popSize):
            #             if terminateds[d] or truncateds[d]:
            #                 dones[d]=0

            #     if np.mean(tfits)>bestFitness: 
            #         with open("best_fit.pkl", "wb") as f:
            #             pickle.dump(testPop[0], f)
            #         bestFitness=np.mean(tfits)
           
            # bestFitCurve[g] = bestFitness
            # print("best fit:",bestFitness)

                


            popVecs = es.ask()
            pop = []        
            for i in range(popSize):
                net = ANN(netSize)
                net.setParamVec(popVecs[i])
                pop.append(net)


        data[run,:] = bestFitCurve
        print(es.result()[1])
        envs.close()
        saveNet = ANN(netSize)
        saveNet.setParamVec(es.result()[0])
        with open("best_net.pkl", "wb") as f:
            pickle.dump(saveNet, f)

    np.savetxt("data\ANN_OpenES_BW.txt",data)


if __name__ == "__main__":
    main()


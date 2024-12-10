from PyANN import ANN
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#14 S/s

def main():

    #Search Parameters
    popSize = 10
    gens = 500
    netSize = [8,32,32,2]
    fitnetSize = [8,32,32,1]
    numSteps = 500

    numMutPoints = int(pow(sum(netSize),0.6))
    print("numMutPoints:",numMutPoints)

    agentsData = np.zeros(gens)
    fitsData = np.zeros(gens)

    for run in range(1):
        print("Run:",run)


        # envs = gym.vector.make("Ant-v4",exclude_current_positions_from_observation=False,num_envs = popSize)
        envs = gym.vector.make("LunarLander-v2",continuous=True,num_envs=popSize)
        # envs = gym.make_vec("BipedalWalker-v3",hardcore=False,num_envs=popSize)

        # Initializing Population
        agentPop = []
        fitPop = []

        for i in range(popSize):
            net = ANN(netSize)
            net.mutateSimple(1)
            fitnet = ANN(fitnetSize)
            fitnet.mutateSimple(1)
            agentPop.append([net,0])
            fitPop.append([fitnet,0])


        #Running gens
        for g in range(gens):
            print("progress:",str(100*g/float(gens))+"%")

            #Main Evaluation

            observations, infos = envs.reset()
            fits = np.zeros((popSize,popSize))
            dones = np.ones(popSize)
            
            for _ in range(numSteps):

                actions = []
                for i in range(popSize):
                    action = agentPop[i][0].step(observations[i])
                    actions.append(action)
                    for j in range(popSize):
                        fits[i,j]+=dones[i]*fitPop[j][0].step(observations[i])[0]
                observations, rewards, terminateds, truncateds, infos = envs.step(actions)

                for d in range(popSize):
                    if terminateds[d] or truncateds[d] or observations[d][1]>1.5:
                        dones[d]=0


            for p in range(popSize):
                agentPop[p][1] = np.mean(fits[p,:])
                fitPop[p][1]=np.std(fits[:,p])

            

            agentPop.sort(key= lambda x : -x[1])
            fitPop.sort(key= lambda x : -x[1])

            agentsData[g] = sum(x[1] for x in agentPop)/popSize
            fitsData[g] = sum(x[1] for x in fitPop)/popSize

            print(agentsData[g],fitsData[g])
            if g%10==0:
                with open("ArbitFit/best_net.pkl", "wb") as f:
                    pickle.dump(agentPop[0][0], f)
                with open("ArbitFit/fit_net.pkl", "wb") as f:
                    pickle.dump(fitPop[0][0], f)


            # Building Pop for next gen

            mutRate =0.1
            
            newFitPop = []
            newFitPop.append([fitPop[0][0],0])
            for i in range(popSize-1):
                b = np.random.randint(0,popSize//2)
                newFitNet = ANN.copy(fitPop[b][0])

                for m in range(numMutPoints):
                    newFitNet.mutateModular(mutRate)
                
                newFitPop.append([newFitNet,0])
            fitPop=newFitPop


            newAgentPop = []  
            newAgentPop.append([agentPop[0][0],0])
            for i in range(popSize-1):
                a = np.random.randint(0,popSize//2)
                newAgentNet = ANN.copy(agentPop[a][0])

                for m in range(numMutPoints):
                    newAgentNet.mutateModular(mutRate)

                newAgentPop.append([newAgentNet,0])
            agentPop=newAgentPop

            

        #End Run  
        with open("ArbitFit/best_net.pkl", "wb") as f:
            pickle.dump(agentPop[0][0], f)
        with open("ArbitFit/fit_net.pkl", "wb") as f:
            pickle.dump(fitPop[0][0], f)

        envs.close()

    plt.plot(agentsData)
    plt.plot(fitsData)
    plt.show()



if __name__ == "__main__":
    main()





    





        
            
    

                


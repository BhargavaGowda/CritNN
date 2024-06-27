from PyCTRNNv3 import CTRNN
import pickle
import copy
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#14 S/s

def main():

    #Search Parameters
    batchSize = 50
    gens = 1000
    netSize = 30
    numSteps = 500

    numRuns = 1
    data = np.zeros((numRuns,gens))
    archiveSize = 250
    archive = []
    # [net,x,y,fit]

    for run in range(numRuns):
        print("Run:",run)


        bestFitness = -10000
        bestFitCurve = np.zeros(gens)
        # envs = gym.vector.make("HalfCheetah-v4",num_envs = batchSize)
        # envs = gym.vector.make("LunarLander-v2",continuous=True,num_envs=batchSize)
        envs = gym.make_vec("BipedalWalker-v3",num_envs = batchSize)

        # Initializing batchulation
        inps = envs.observation_space.shape[1]
        outs = envs.action_space.shape[1]

        batch = []
        for i in range(batchSize):
            net = CTRNN(netSize,10,10)
            net.mutateSimple(2)
            net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
            net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
            batch.append(net)


        #Running gens
        for g in range(gens):
            print("progress:",str(100*g/float(gens))+"%")

            #Main Evaluation

            observations, infos = envs.reset()
            fits = np.zeros(batchSize)
            dones = np.ones(batchSize)
            for i in range(batchSize):
                batch[i].reset()

            for _ in range(numSteps):

                actions = []
                for i in range(batchSize):
                    action = batch[i].step(np.concatenate((observations[i],np.zeros(net.size-inps))))[-outs:]
                    actions.append(action)

                observations, rewards, terminateds, truncateds, infos = envs.step(actions)
                fits += dones*rewards

                for d in range(batchSize):
                    if terminateds[d] or truncateds[d]:
                        dones[d]=0

            for p in range(batchSize):
                
                #ARCHIVE STUFF
                x = batch[p].bias[-1]
                y = batch[p].bias[-2]

                x = int(np.clip(x*(archiveSize/20)+archiveSize//2,0,archiveSize-1))
                y = int(np.clip(y*(archiveSize/20)+archiveSize//2,0,archiveSize-2))

                newCell = True
                for a in archive:
                    if a[1]==x and a[2]==y:
                        newCell=False
                        if fits[p]>a[3]:
                            a[0] = batch[p]
                            a[3] = fits[p]                          
                            break
                if newCell:
                    archive.append([batch[p],x,y,fits[p]])

            archive.sort(key= lambda x: -x[3])
            bestFitCurve[g] = archive[0][3]
            print("best fit:",bestFitCurve[g])


            # Building batch for next gen
            newbatch = []
            for i in range(batchSize):

                a = np.random.randint(0,len(archive))

                newNet = CTRNN.recombine(archive[a][0],archive[a][0])

                mutRate =0.1
                newNet.mutateSimple(mutRate)

                newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
                newNet.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))

                newbatch.append(newNet)

            batch = newbatch

            if g%100 ==0 and g>0:
                with open("ArchiveEvo/archive.pkl", "wb") as f:
                    pickle.dump(archive, f)
                    print("saved at",g)

        #End Run  
        with open("ArchiveEvo/archive.pkl", "wb") as f:
            pickle.dump(archive, f)
        data[run,:] = bestFitCurve
        print(bestFitCurve[-1])
        envs.close()

    np.savetxt("data/MAGEvoResults.txt",data)
    


if __name__ == "__main__":
    main()





    





        
            
    

                


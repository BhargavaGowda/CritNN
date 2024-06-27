from PyCTRNNv3 import CTRNN
import pickle
import copy
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#14 S/s

def main():

    #Search Parameters
    batchSize = 50
    maxBatches = 20000#gens
    netSize = 30
    numSteps = 500
    numRuns = 1
    data = np.zeros((numRuns,maxBatches))
    archiveInitSize = 200
    mutRate = 5.0/np.sqrt(archiveInitSize)
    lerpFactor=0.7/archiveInitSize
    pici = 0
    

    for run in range(numRuns):
        print("Run:",run)

        batchNum = 0
        bestFitCurve = np.zeros(maxBatches)
        # envs = gym.make_vec("Hopper-v4",num_envs = batchSize)
        # envs = gym.make_vec("LunarLander-v2",continuous=True,num_envs=batchSize)
        envs = gym.make_vec("BipedalWalker-v3",num_envs = batchSize)

        
        inps = envs.observation_space.shape[1]
        outs = envs.action_space.shape[1]

        # Initializing archive
        archive = []
        archiveIndex = 0
        # [centroid net,elite net,fit]
        for i in range(archiveInitSize):
            centroidNet = CTRNN(netSize,10,10)
            centroidNet.mutateSimple(5)
            centroidNet.setInputs(np.concatenate([np.ones(inps),np.zeros(netSize-inps)]))
            centroidNet.setOutputs(np.concatenate([np.zeros(netSize-outs),np.ones(outs)]))
            centroidNet.reset()
            archive.append([centroidNet,None,-np.inf])




        #Running gens
        while batchNum<maxBatches:
            print("progress:",str(100*batchNum/float(maxBatches))+"%")

            #Prep next Batch
            batch = []
            for i in range(batchSize):
                a = np.random.randint(0,len(archive))
                newNet= CTRNN.copy(archive[a][0])
                newNet.mutateSimple(mutRate)
                newNet.setInputs(np.concatenate([np.ones(inps),np.zeros(netSize-inps)]))
                newNet.setOutputs(np.concatenate([np.zeros(netSize-outs),np.ones(outs)]))
                newNet.reset()
                batch.append(newNet)

            #Main Evaluation
            observations, infos = envs.reset()
            fits = np.zeros(batchSize)
            dones = np.ones(batchSize)

            for _ in range(numSteps):

                actions = []
                for i in range(batchSize):
                    action = batch[i].step(np.concatenate((observations[i],np.zeros(netSize-inps))))[-outs:]
                    actions.append(action)

                observations, rewards, terminateds, truncateds, infos = envs.step(actions)
                fits += dones*rewards

                for d in range(batchSize):
                    if terminateds[d] or truncateds[d]:
                        dones[d]=0
            for p in range(batchSize):
                closestCentroid = None
                closestDist = np.inf
                for arch in range(len(archive)):
                    dist = CTRNN.getDistance(batch[p],archive[arch][0])
                    if dist<closestDist:
                        closestCentroid = arch
                        closestDist = dist

                if fits[p]>archive[closestCentroid][2]:
                    archive[closestCentroid][1] = batch[p]
                    archive[closestCentroid][2] = fits[p]
                
            
            # Updating Centroids
            filledCells = [c for c in archive if c[1]]
            for c in filledCells:
                c[0].lerpNet(c[1],lerpFactor)
                

            bestFit = np.max([c[2] for c in archive])
            bestFitCurve[batchNum] = bestFit
            print("best fit:",bestFit)


            if batchNum%50 == 0 and batchNum>0:
                x, y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
                points = np.array([[c[0].bias[-1],c[0].bias[-2]] for c in archive])
                values = [c[2] for c in archive]#np.linspace(0,1,len(archive))
                map = griddata(points,values,(x,y),method='nearest')
                cmap = copy.copy(plt.get_cmap('RdBu'))
                cmap.set_bad('black',1.)
                plt.imshow(map,cmap=cmap,vmin=-20,vmax=50,origin = "lower",interpolation='none')
                plt.colorbar()
                plt.scatter(50*points[:,0]+500,50*points[:,1]+500)
                points = np.array([[c[1].bias[-1],c[1].bias[-2]] for c in archive if c[1]])
                plt.scatter(50*points[:,0]+500,50*points[:,1]+500)
                plt.savefig('figs/CVTFigs/CVT_'+str(pici)+'.png', bbox_inches='tight')
                plt.clf()
                pici+=1
            
            if batchNum%1000 == 0 and batchNum>0:
                with open("ArchiveEvo/CVTarchive_checkpoint_"+str(batchNum)+".pkl", "wb") as f:
                    pickle.dump(archive, f)
                    print("saved at",batchNum)
            batchNum+=1


        #End Run  
        with open("ArchiveEvo/CVTarchiveBPWalker.pkl", "wb") as f:
            pickle.dump(archive, f)
        data[run,:] = bestFitCurve
        print(bestFitCurve[-1])
        envs.close()

    np.savetxt("data/CVTMAGEBPWalker.txt",data)
    


if __name__ == "__main__":
    main()





    





        
            
    

                


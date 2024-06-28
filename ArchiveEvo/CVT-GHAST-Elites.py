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
    maxBatches = 200#gens
    netSize = 25
    numSteps = 500
    numRuns = 1
    data = np.zeros((numRuns,maxBatches))
    archiveInitSize = 100
    mutRate = 5.0/np.sqrt(archiveInitSize)
    lerpFactor=0.7/archiveInitSize
    pici = 0
    

    for run in range(numRuns):
        print("Run:",run)

        batchNum = 0
        bestFitCurve = np.zeros(maxBatches)
        envs = gym.make_vec("HalfCheetah-v4",num_envs = batchSize)
        # envs = gym.make_vec("LunarLander-v2",continuous=True,num_envs=batchSize)
        # envs = gym.make_vec("BipedalWalker-v3",num_envs = batchSize)

        
        inps = envs.observation_space.shape[1]
        outs = envs.action_space.shape[1]

        # Initializing archive
        archive = []
        archiveIndex = 0
        # [centroid net,trajectory,elite,fit]
        for i in range(archiveInitSize):
            centroidNet = CTRNN(netSize,10,10)
            centroidNet.mutateSimple(5)
            centroidNet.setInputs(np.concatenate([np.ones(inps),np.zeros(netSize-inps)]))
            centroidNet.setOutputs(np.concatenate([np.zeros(netSize-outs),np.ones(outs)]))
            centroidNet.reset()
            archive.append([centroidNet,None,None,-np.inf])




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

            trajectories = np.zeros((batchSize,numSteps,inps))
            for _ in range(numSteps):
                for d in range(batchSize):
                    trajectories[d,_,:]=observations[d]*dones[d]

                actions = []
                for i in range(batchSize):
                    action = batch[i].step(np.concatenate((observations[i],np.zeros(netSize-inps))))[-outs:]
                    actions.append(action)

                observations, rewards, terminateds, truncateds, infos = envs.step(actions)
                fits += dones*rewards

                for d in range(batchSize):
                    if terminateds[d] or truncateds[d]:
                        dones[d]=0

            #Assign Trajectories  
            for p in range(batchSize):
                closestCentroid = None
                closestDist = np.inf
                for arch in range(len(archive)):
                    dist = CTRNN.getDistance(batch[p],archive[arch][0])
                    if dist<closestDist:
                        closestCentroid = arch
                        closestDist = dist

                if fits[p]>archive[closestCentroid][3]:
                    archive[closestCentroid][2] = batch[p]
                    archive[closestCentroid][3] = fits[p]

                archive[closestCentroid][1] = np.copy(trajectories[p,:,:])

                
            
            # Updating Centroids
            filledCells = [c for c in archive if c[2]]
            for c in filledCells:
                farthestTrajectory = None
                farthestTDist = 0
                for c2 in filledCells:
                    if c!=c2:
                        dist = np.linalg.norm(c[1]-c2[1])
                        if dist>farthestTDist:
                            farthestTDist=dist
                            farthestTrajectory=c2

                c[0].lerpNet(farthestTrajectory[0],1/CTRNN.getDistance(c[0],farthestTrajectory[0]))
            for c in filledCells:
                closestTrajectory = None
                closestTDist = np.inf
                for c2 in filledCells:
                    if c!=c2:
                        dist = np.linalg.norm(c[1]-c2[1])
                        if dist<closestTDist:
                            closestTDist=dist
                            closestTrajectory=c2

                c[0].lerpNet(closestTrajectory[0],-1/CTRNN.getDistance(c[0],closestTrajectory[0]))

        

            bestFit = np.max([c[3] for c in archive])
            bestFitCurve[batchNum] = bestFit
            print("best fit:",bestFit)


            if batchNum%5 == 0 and batchNum>0:
                # x, y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
                # points = np.array([[c[0].bias[0],c[0].bias[1]] for c in archive])
                # values = np.linspace(0,1,len(archive))#[c[2] for c in archive]
                # map = griddata(points,values,(x,y),method='nearest')

                points = np.array([[c[1][400,2],c[1][400,3]] for c in filledCells])
                values = np.linspace(0,1,len(filledCells))
                plt.scatter(points[:,0],points[:,1],c=values,cmap="Set3")
                plt.savefig('figs/CVTFigs/CVTGHAST_'+str(pici)+'.png')#, bbox_inches='tight')
                plt.clf()
                pici+=1
            
            # if batchNum%1000 == 0 and batchNum>0:
            #     with open("ArchiveEvo/CVTarchive_checkpoint_"+str(batchNum)+".pkl", "wb") as f:
            #         pickle.dump(archive, f)
            #         print("saved at",batchNum)
            batchNum+=1


        #End Run  
        with open("ArchiveEvo/CVT-GHAST.pkl", "wb") as f:
            pickle.dump(archive, f)
        data[run,:] = bestFitCurve
        print(bestFitCurve[-1])
        envs.close()

    np.savetxt("data/CVTGHAST.txt",data)
    


if __name__ == "__main__":
    main()





    





        
            
    

                


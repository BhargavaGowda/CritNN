from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LogNorm

# env = gym.make("InvertedDoublePendulum-v4")
env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")
# env = gym.make("BipedalWalker-v3",hardcore=True)
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]

# net = CTRNN(10)
# net.mutateSplit(3,1)
# net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
# net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

steps = 40
for z in range(steps):
    print("progress:",str(100*z/float(steps))+"%")
    with open("net.pkl", "rb") as f:
        net = pickle.load(f)
        net.reset()

    imgSize = 50
    paramStep = 0.01#1.6/pow(1.4,z)

    i_off=0.02
    j_off=0.004

    b1=-1
    b2=-2
    w1=[11,6]
    w2=[11,7]

    imgfit = np.zeros((imgSize,imgSize))
    imgObs=np.zeros((imgSize,imgSize,inps))
    imgPots=np.zeros((imgSize,imgSize,net.size))

    net.weights[w1[0],w1[1]] +=i_off
    net.weights[w2[0],w2[1]] +=j_off
    a=float(net.weights[w1[0],w1[1]])
    b=float(net.weights[w2[0],w2[1]])
    # net.bias[b1] += i_off
    # net.bias[b2] += j_off
    # a = float(net.bias[b1])
    # b = float(net.bias[b2])
    print(a,b)

    net.weights[w1[0],w1[1]] += -0.5*paramStep*imgSize
    net.weights[w2[0],w2[1]] += -0.5*paramStep*imgSize
    # net.bias[b1] += -0.5*paramStep*imgSize
    # net.bias[b2] += -0.5*paramStep*imgSize


    for i in range(imgSize):
        
        for j in range(imgSize):
            net.reset()
            # SEED
            observation, info = env.reset(seed=4)
            env.set_state(np.array([0,0,-0.05+0.0025*z]),np.array([0,0,0]))
            fitness = 0
            while True:

                inp = np.array(observation)
                action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
                observation, reward, terminated, truncated, info = env.step(action[-outs:])
                fitness+= reward

                if terminated or truncated:
                    break

            # img[j,i] = observation[0]
            imgfit[j,i] = fitness
            imgObs[j,i] = observation
            imgPots[j,i] = net.potentials
            fitness = 0

            net.weights[w2[0],w2[1]]+=paramStep
        net.weights[w1[0],w1[1]]+=paramStep
        net.weights[w2[0],w2[1]]-=paramStep*imgSize

        #     net.bias[b2]+=paramStep
        # net.bias[b1]+=paramStep
        # net.bias[b2]-=paramStep*imgSize


    # print("Saving images...")

    # plt.xlabel("bias:"+str(b1))
    # plt.ylabel("bias:"+str(b2))
    plt.xlabel("weight:"+str(w1[0])+","+str(w1[1]))
    plt.ylabel("weight:"+str(w2[0])+","+str(w2[1]))
    plt.imshow(imgfit,norm=LogNorm(vmin=50,vmax = 5000),cmap="RdYlGn",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
    plt.colorbar()
    plt.title("Fitness")
    plt.savefig('movie/step_'+str(z)+'.png', bbox_inches='tight')
    plt.clf()


# for ob in range(inps):
#     plt.imshow(imgObs[:,:,ob],cmap="pink",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
#     plt.colorbar()
#     plt.title("Observation "+str(ob))
#     # plt.xlabel("bias:"+str(b1))
#     # plt.ylabel("bias:"+str(b2))
#     plt.xlabel("weight:"+str(w1[0])+","+str(w1[1]))
#     plt.ylabel("weight:"+str(w2[0])+","+str(w2[1]))
#     plt.savefig('figs/obs_'+str(ob)+'.png', bbox_inches='tight')
#     plt.clf()

# for pot in range(net.size):
#     plt.imshow(imgPots[:,:,pot],cmap="bone",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
#     plt.colorbar()
#     # plt.xlabel("bias:"+str(b1))
#     # plt.ylabel("bias:"+str(b2))
#     plt.xlabel("weight:"+str(w1[0])+","+str(w1[1]))
#     plt.ylabel("weight:"+str(w2[0])+","+str(w2[1]))
#     plt.title("Potential "+str(pot))
#     plt.savefig('figs/pots_'+str(pot)+'.png', bbox_inches='tight')
#     plt.clf()

print("Done.")


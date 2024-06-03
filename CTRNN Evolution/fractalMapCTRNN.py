from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
# env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")
env = gym.make("BipedalWalker-v3",hardcore=True)
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]

# net = CTRNN(10)
# net.mutateSplit(3,1)
# net.setInputs(np.concatenate([np.ones(inps),np.zeros(net.size-inps)]))
# net.setOutputs(np.concatenate([np.zeros(net.size-outs),np.ones(outs)]))
# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

with open("best_net.pkl", "rb") as f:
    net = pickle.load(f)
    net.reset()
print(net.bias)
imgSize = 50
paramStep = 0.1

i_off=-2.5+0.31
j_off=0

p1=13
p2=16



imgfit = np.zeros((imgSize,imgSize))
img0 = np.zeros((imgSize,imgSize))
img1 = np.zeros((imgSize,imgSize))
img2 = np.zeros((imgSize,imgSize))

# net.weights[net.size-2,2] +=i_off
# net.weights[net.size-1,2] +=j_off
# a=float(net.weights[net.size-2,2])
# b=float(net.weights[net.size-1,2])
net.bias[p1] += i_off
net.bias[p2] += j_off
a = float(net.bias[p1])
b = float(net.bias[p2])



print(net.size)
print(a,b)

# net.weights[net.size-2,2] += -0.5*paramStep*imgSize
# net.weights[net.size-1,2] += -0.5*paramStep*imgSize
net.bias[p1] += -0.5*paramStep*imgSize
net.bias[p2] += -0.5*paramStep*imgSize


for i in range(imgSize):
    print("progress:",str(100*i/float(imgSize))+"%")
    for j in range(imgSize):
        net.reset()
        observation, info = env.reset(seed=3)
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
        img0[j,i] = observation[0]
        img1[j,i] = observation[1]
        img2[j,i] = observation[2]
        fitness = 0

        net.bias[p2]+=paramStep
    net.bias[p1]+=paramStep
    net.bias[p2]-=paramStep*imgSize



fig, axs = plt.subplots(2,2)
axs = axs.flatten()
im = axs[0].imshow(imgfit,cmap="magma",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[0])
im = axs[1].imshow(img0,cmap="YlGn_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[1])
im = axs[2].imshow(img1,cmap="RdPu_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[2])
im = axs[3].imshow(img2,cmap="GnBu_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[3])
fig.supxlabel("bias:"+str(p1))
fig.supylabel("bias:"+str(p2))

#titles
axs[0].title.set_text("Fitness")
axs[1].title.set_text("x")
axs[2].title.set_text("y")
axs[3].title.set_text("x speed")

#labels


plt.show()
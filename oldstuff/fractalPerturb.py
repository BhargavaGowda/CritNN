from PyCTRNNv3 import CTRNN
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0")
env = gym.make("LunarLander-v2",continuous=True)
# env = gym.make("BipedalWalker-v3")


inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]

# net = CTRNN(30)
# net.mutateSimple(3)
# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

with open("net.pkl", "rb") as f:
    net = pickle.load(f)
    net.reset()

print(net.size)

imgSize = 50
paramStep = 0.1

i_off=0
j_off=0


img0 = np.zeros((imgSize,imgSize))
img1 = np.zeros((imgSize,imgSize))
img2 = np.zeros((imgSize,imgSize))
img3 = np.zeros((imgSize,imgSize))
img4 = np.zeros((imgSize,imgSize))
img5 = np.zeros((imgSize,imgSize))
img6 = np.zeros((imgSize,imgSize))
img7 = np.zeros((imgSize,imgSize))
img8 = np.zeros((imgSize,imgSize))
img9 = np.zeros((imgSize,imgSize))
a=0
b=0

seed = np.random.randint(0,100000)
# seed = 4
print("seed:",seed)
    


for i in range(imgSize):
    print("progress:",str(100*i/float(imgSize))+"%")
    for j in range(imgSize):
        net.reset()
        observation, info = env.reset(seed=seed)
        fitness = 0
        for _ in range(400):

            inp = np.array(observation)
            action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
            observation, reward, terminated, truncated, info = env.step(action[-outs:])
            fitness+= reward

            if _==5:

                net.potentials[0]+=(i-imgSize//2)*paramStep
                net.potentials[1]+=(j-imgSize//2)*paramStep

            if terminated or truncated:
                break


        # img0[j,i] = net.potentials[0]
        # img1[j,i] = net.potentials[1]
        # img2[j,i] = net.potentials[2]
        # img3[j,i] = net.potentials[3]
        # img4[j,i] = net.potentials[4]
        # img5[j,i] = net.potentials[5]
        # img6[j,i] = net.potentials[6]
        # img7[j,i] = net.potentials[7]
        # img8[j,i] = net.potentials[8]
        # img9[j,i] = net.potentials[9]

        img0[j,i] = observation[0]
        img1[j,i] = observation[1]
        img2[j,i] = observation[2]
        img3[j,i] = observation[3]
        img4[j,i] = observation[4]
        img5[j,i] = observation[5]
        img6[j,i] = observation[6]
        img7[j,i] = fitness
        # img8[j,i] = observation[8]
        # img9[j,i] = observation[9]

        fitness = 0




fig, axs = plt.subplots(2,5)
axs = axs.flatten()
im = axs[0].imshow(img0,cmap="BrBG",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[0])
im = axs[1].imshow(img1,cmap="BrBG",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[1])
im = axs[2].imshow(img2,cmap="PiYG",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[2])
im = axs[3].imshow(img3,cmap="PiYG",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[3])
im = axs[4].imshow(img4,cmap="PRGn_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[4])
im = axs[5].imshow(img5,cmap="PRGn_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[5])
im = axs[6].imshow(img6,cmap="PuOr",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[6])
im = axs[7].imshow(img7,cmap="PuOr",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[7])
im = axs[8].imshow(img8,cmap="Spectral",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[8])
im = axs[9].imshow(img9,cmap="Spectral",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[9])

# plt.imshow(imgfit,cmap="magma",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)]
#                ,origin = "lower",interpolation='none')#,vmin=-0.2,vmax=0.2)
# plt.colorbar()

#titles
axs[0].title.set_text("x")
axs[1].title.set_text("vel")
axs[2].title.set_text("~")
axs[3].title.set_text("~")
axs[4].title.set_text("~")
axs[5].title.set_text("~")
axs[6].title.set_text("~")
axs[7].title.set_text("~")
axs[8].title.set_text("~")
axs[9].title.set_text("~")

# axs[0].title.set_text("pots 0")
# axs[1].title.set_text("pots 1")
# axs[2].title.set_text("pots 2")
# axs[3].title.set_text("pots 3")
# axs[4].title.set_text("pots 4")
# axs[5].title.set_text("pots 5")
# axs[6].title.set_text("pots 6")
# axs[7].title.set_text("pots 7")
# axs[8].title.set_text("pots 8")
# axs[9].title.set_text("pots 9")

#labels


plt.show()
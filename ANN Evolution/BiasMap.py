from PyANN import ANN
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LogNorm


# env = gym.make("HalfCheetah-v4")
# env = gym.make("LunarLander-v2",continuous=True)
env = gym.make("BipedalWalker-v3")
# env = gym.make("BipedalWalker-v3",hardcore=True)

net = ANN([24,40,4])
net.mutateSimple()
with open("net.pkl", "wb") as f:
    pickle.dump(net, f)

# with open("best_net.pkl", "rb") as f:
#     net = pickle.load(f)
#     net.reset()

print(net.size)

imgSize = 50
paramStep = 0.2

i_off=0
j_off=0

b1=-1
b2=-2
w1=[1,1,17]
w2=[1,1,18]

imgfit = np.zeros((imgSize,imgSize))


net.weights[w1[0]][w1[1],w1[2]] +=i_off
net.weights[w2[0]][w2[1],w2[2]] +=j_off
a=float(net.weights[w1[0]][w1[1],w1[2]])
b=float(net.weights[w2[0]][w2[1],w2[2]])
# net.bias[b1] += i_off
# net.bias[b2] += j_off
# a = float(net.bias[b1])
# b = float(net.bias[b2])
print(a,b)

net.weights[w1[0]][w1[1],w1[2]] += -0.5*paramStep*imgSize
net.weights[w2[0]][w2[1],w2[2]] += -0.5*paramStep*imgSize
# net.bias[b1] += -0.5*paramStep*imgSize
# net.bias[b2] += -0.5*paramStep*imgSize


for i in range(imgSize):
    print("progress:",str(100*i/float(imgSize))+"%")
    for j in range(imgSize):
        # SEED
        observation, info = env.reset(seed=4)
        fitness = 0
        while True:

            inp = np.array(observation)
            action = net.step(inp)
            observation, reward, terminated, truncated, info = env.step(action)
            fitness+= reward

            if terminated or truncated:
                break

        imgfit[j,i] = fitness
        fitness = 0

        net.weights[w2[0]][w2[1],w2[2]]+=paramStep
    net.weights[w1[0]][w1[1],w1[2]]+=paramStep
    net.weights[w2[0]][w2[1],w2[2]]-=paramStep*imgSize

    #     net.bias[b2]+=paramStep
    # net.bias[b1]+=paramStep
    # net.bias[b2]-=paramStep*imgSize


print("Saving images...")

# plt.xlabel("bias:"+str(b1))
# plt.ylabel("bias:"+str(b2))
plt.xlabel("weight:"+str(w1[0])+","+str(w1[1]))
plt.ylabel("weight:"+str(w2[0])+","+str(w2[1]))
plt.imshow(imgfit,cmap="RdYlGn",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar()
plt.title("Fitness")
plt.savefig('figs/fitness.png', bbox_inches='tight')


# fig, axs = plt.subplots(2,2)
# axs = axs.flatten()
# im = axs[0].imshow(imgfit,cmap="magma",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
# plt.colorbar(im,ax=axs[0])
# im = axs[1].imshow(img0,cmap="YlGn_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
# plt.colorbar(im,ax=axs[1])
# im = axs[2].imshow(img1,cmap="RdPu_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
# plt.colorbar(im,ax=axs[2])
# im = axs[3].imshow(img2,cmap="GnBu_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
# plt.colorbar(im,ax=axs[3])
# fig.supxlabel("bias:"+str(b1))
# fig.supylabel("bias:"+str(b2))

# #titles
# axs[0].title.set_text("Fitness")
# axs[1].title.set_text("-1")
# axs[2].title.set_text("-2")
# axs[3].title.set_text("-3")

# #labels


plt.show()
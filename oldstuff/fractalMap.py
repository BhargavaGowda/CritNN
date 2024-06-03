import torch
from torch import nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True)
torch.set_grad_enabled(False)

model = nn.Sequential(
    nn.Linear(8,4),
    nn.Tanh(),
    nn.Linear(4,2),
    nn.Tanh()

)

torch.save(model,"base model.modl")
# model = torch.load("bestLander.modl")

a=0
b=0

imgSize = 500
imgfit = np.zeros((imgSize,imgSize))
img0 = np.zeros((imgSize,imgSize))
img1 = np.zeros((imgSize,imgSize))
img2 = np.zeros((imgSize,imgSize))
paramStep = 0.01

i_off=+4
j_off=-4

for i in range(imgSize):
    print("progress:",str(100*i/float(imgSize))+"%")
    for j in range(imgSize):

        model = torch.load("bestLander.modl")

        # param stepping
        params = []
        for name,param in model.named_parameters():
            params.append(param)

        params[2].data[0,3]+=paramStep*(i-imgSize//2)+i_off
        params[2].data[1,3]+=paramStep*(j-imgSize//2)+j_off

        observation, info = env.reset(seed=2)
        fitness = 0
        while True:

            inp = np.array(observation,dtype=np.float32)
            inp = torch.from_numpy(inp)
            output = model(inp)
            action = output.detach().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            fitness+= reward

            if terminated or truncated:
                break

        # img[j,i] = observation[0]
        imgfit[j,i] = fitness
        img0[j,i] = observation[2]
        img1[j,i] = observation[3]
        img2[j,i] = observation[5]
        fitness = 0


model = torch.load("bestLander.modl")
params = []
for name,param in model.named_parameters():
    params.append(param)
a=params[2].data[0,3]+i_off
b=params[2].data[1,3]+j_off
print(a,b)

fig, axs = plt.subplots(2,2)
axs = axs.flatten()
im = axs[0].imshow(imgfit,cmap="RdBu",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[0])
im = axs[1].imshow(img0,cmap="YlGn_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[1])
im = axs[2].imshow(img1,cmap="RdPu_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[2])
im = axs[3].imshow(img2,cmap="GnBu_r",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)],origin = "lower",interpolation='none')
plt.colorbar(im,ax=axs[3])

#titles
axs[0].title.set_text("Fitness")
axs[1].title.set_text("x vel")
axs[2].title.set_text("y vel")
axs[3].title.set_text("angle vel")

#labels


plt.show()
            
                


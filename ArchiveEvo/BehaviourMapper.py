import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
from PyCTRNNv3 import CTRNN
from scipy.interpolate import griddata
import copy

with open("ArchiveEvo\CVTarchive.pkl", "rb") as f:
    archive = pickle.load(f)
# env = gym.make("HalfCheetah-v4",render_mode="human")
# env = gym.make("LunarLander-v2",continuous=True,render_mode="human")
env = gym.make("BipedalWalker-v3")
# env = gym.make("BipedalWalker-v3",hardcore=True,render_mode="human")
inps = env.observation_space.shape[0]
outs = env.action_space.shape[0]



behaviours=[]
for i in range(len(archive)): 
    observation, info = env.reset()
    net = archive[i][0]
    net.reset()

    obs=[]
    for _ in range(500):

        inp = np.array(observation)
        action = net.step(np.concatenate((inp,np.zeros(net.size-inps))))
        # action = np.zeros(outs)
        observation, reward, terminated, truncated, info = env.step(action[-outs:])
        obs.append(observation[2])


        if terminated or truncated:
            break
    behaviours.append(np.mean(obs))


x, y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))

#Centroids
points = np.array([[c[1].bias[-1],c[1].bias[-2]] for c in archive])
values = behaviours
map = griddata(points,values,(x,y),method='nearest')


cmap = copy.copy(plt.get_cmap('cividis'))
cmap.set_bad('black',1.)
plt.imshow(map,cmap=cmap,origin = "lower",interpolation='none')
plt.colorbar()
plt.scatter(50*points[:,0]+500,50*points[:,1]+500)
# plt.scatter(x,y)
plt.savefig('figs/CVT_4.png', bbox_inches='tight')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PyCTRNNv3 import CTRNN
from scipy.interpolate import griddata
import copy

with open("ArchiveEvo\CVTarchive.pkl", "rb") as f:
    archive = pickle.load(f)





print(len(archive))
# print(sum([CTRNN.getDistance(archive[0][0],c[0]) for c in archive])/len(archive))

x, y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))

# #Elites-Fitness
# points = np.array([[c[1].bias[0],c[1].bias[1]] for c in archive if c[1]])
# values = [c[2] for c in archive if c[1]]
# map = griddata(points,values,(x,y),method='linear')

#Centroids
points = np.array([[c[0].bias[2],c[0].bias[3]] for c in archive])
# values = np.linspace(0,len(archive)-1,len(archive))
values = [c[2] for c in archive]
map = griddata(points,values,(x,y),method='nearest')




cmap = copy.copy(plt.get_cmap('magma'))
cmap.set_bad('black',1.)
plt.imshow(map,cmap=cmap,vmin=-50,vmax=100,origin = "lower",interpolation='none')
plt.colorbar()

with open("best_fit.pkl", "rb") as f:
    bestNet = pickle.load(f)

selectedCentroid = np.random.randint(len(archive))
mutated = []
for i in range(100):
    mutNet = CTRNN.copy(archive[selectedCentroid][0])
    mutNet.mutateSimple(5.0/np.sqrt(len(archive)))
    mutated.append(mutNet)


points = np.array([[m.bias[2],m.bias[3]] for m in mutated])

plt.scatter(50*points[:,0]+500,50*points[:,1]+500)
plt.scatter(500+50*archive[selectedCentroid][0].bias[2],500+50*archive[selectedCentroid][0].bias[3])
plt.savefig('figs/CVT_4.png', bbox_inches='tight')
plt.show()
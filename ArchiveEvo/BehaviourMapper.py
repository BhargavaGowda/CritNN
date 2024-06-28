import numpy as np
import matplotlib.pyplot as plt
import pickle
from PyCTRNNv3 import CTRNN
from scipy.interpolate import griddata
import copy

with open("ArchiveEvo\CVT-GHAST.pkl", "rb") as f:
    archive = pickle.load(f)





print(len(archive))
# print(sum([CTRNN.getDistance(archive[0][0],c[0]) for c in archive])/len(archive))

x, y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))

#Centroids
points = np.array([[c[1][400,2],c[1][400,3]] for c in archive])
# values = np.linspace(0,len(archive)-1,len(archive))
# # values = [c[2] for c in archive]
# map = griddata(points,values,(x,y),method='nearest')




cmap = copy.copy(plt.get_cmap('tab10'))
cmap.set_bad('black',1.)
# plt.imshow(map,cmap=cmap,origin = "lower",interpolation='none')
# plt.colorbar()
plt.scatter(points[:,0],points[:,1])
plt.savefig('figs/CVT_4.png', bbox_inches='tight')
plt.show()
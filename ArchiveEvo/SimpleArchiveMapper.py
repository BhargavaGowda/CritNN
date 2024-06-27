import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

with open("ArchiveEvo/archive.pkl", "rb") as f:
    archive = pickle.load(f)


map = np.full((250,250),np.nan)

print(len(archive))
for i in range(len(archive)):
    map[archive[i][2],archive[i][1]] = archive[i][3]



cmap = copy.copy(plt.get_cmap('inferno'))
cmap.set_under('white',1.)
plt.imshow(map,cmap=cmap,origin = "lower",interpolation='none')
plt.savefig('figs/MAGEliteMap_Random.png', bbox_inches='tight')
plt.show()
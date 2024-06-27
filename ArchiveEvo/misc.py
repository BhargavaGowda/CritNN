import pickle
import numpy as np
import matplotlib.pyplot as plt
from PyCTRNNv3 import CTRNN

# baseNet = CTRNN(10)
# total = 0
# for i in range(300):
#     newNet = CTRNN.copy(baseNet)
#     newNet.mutateSimple(1)
#     total+=CTRNN.getDistance(newNet,baseNet)

# print(total/300)


# data1 = np.loadtxt("data/Cheetah1.txt")
# data2 = np.loadtxt("data/Cheetah2.txt")
data3 = np.loadtxt("data\CVTMAGEvoResults.txt")

# plt.boxplot([data1,data2,data3])
plt.ylabel("Fitness")

# plt.bar(["bad","ok","good"],[np.std(data1),np.std(data2),np.std(data3)])
plt.plot(data3)
plt.title("Fitness distribution across 100 runs")
plt.show()


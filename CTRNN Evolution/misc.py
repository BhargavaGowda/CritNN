import pickle
import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("data/Cheetah1.txt")
data2 = np.loadtxt("data/Cheetah2.txt")
data3 = np.loadtxt("data/Cheetah3.txt")

# plt.boxplot([data1,data2,data3])
plt.ylabel("Fitness")

plt.bar(["bad","ok","good"],[np.std(data1),np.std(data2),np.std(data3)])
plt.title("Fitness distribution across 100 runs")
plt.show()


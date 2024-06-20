import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("data/CritEvoResults.txt")
mean = np.zeros(data1.shape[1])
variance = np.zeros(data1.shape[1])


for i in range(data1.shape[1]):
    mean[i] = np.mean(data1[:,i])
    variance[i] = np.std(data1[:,i])


plt.plot(range(data1.shape[1]),mean)
plt.fill_between(range(data1.shape[1]),mean+variance,mean-variance, color='blue',alpha=0.2)
plt.ylabel("Fitness")
plt.show()


import numpy as np
import matplotlib.pyplot as plt


data1 = np.loadtxt("data\SimpleEvoResults.txt")
data2 = np.loadtxt("data\RecombinantEvoResults.txt")

mean = np.zeros(data1.shape[1])
variance = np.zeros(data1.shape[1])


for i in range(data1.shape[1]):
    mean[i] = np.mean(data1[:,i])
    variance[i] = np.std(data1[:,i])


plt.plot(range(data1.shape[1]),mean)
plt.fill_between(range(data1.shape[1]),mean+variance,mean-variance, color='blue',alpha=0.2)

mean = np.zeros(data2.shape[1])
variance = np.zeros(data2.shape[1])


for i in range(data2.shape[1]):
    mean[i] = np.mean(data2[:,i])
    variance[i] = np.std(data2[:,i])


plt.plot(range(data2.shape[1]),mean)
plt.fill_between(range(data2.shape[1]),mean+variance,mean-variance, color='orange',alpha=0.2)

plt.ylabel("Fitness")
plt.xlabel("Batches")
plt.show()


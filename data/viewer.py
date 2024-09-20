import numpy as np
import matplotlib.pyplot as plt


data1 = np.loadtxt("data\ANN_SimpleEvoModular_BW.txt")


mean = np.zeros(data1.shape[1])
variance = np.zeros(data1.shape[1])


for i in range(data1.shape[1]):
    mean[i] = np.mean(data1[:,i])
    variance[i] = np.std(data1[:,i])


plt.plot(range(data1.shape[1]),mean,color='blue',label='OpenES')
plt.fill_between(range(data1.shape[1]),mean+variance,mean-variance, color='blue',alpha=0.2)


data2 = np.loadtxt("data\ANN_RecombFullModular_BW.txt")
mean = np.zeros(data2.shape[1])
variance = np.zeros(data2.shape[1])


for i in range(data2.shape[1]):
    mean[i] = np.mean(data2[:,i])
    variance[i] = np.std(data2[:,i])


plt.plot(range(data2.shape[1]),mean,color="orange",label='Non-Modular Mutation GA')
plt.fill_between(range(data2.shape[1]),mean+variance,mean-variance, color='orange',alpha=0.2)

# data3 = np.loadtxt("data\ANN_SimpleEvoModular_HfCh.txt")
# mean = np.zeros(data3.shape[1])
# variance = np.zeros(data3.shape[1])


# for i in range(data3.shape[1]):
#     mean[i] = np.mean(data3[:,i])
#     variance[i] = np.std(data3[:,i])


# plt.plot(range(data3.shape[1]),mean,color="green",label='Modular Mutation GA')
# plt.fill_between(range(data3.shape[1]),mean+variance,mean-variance, color='green',alpha=0.2)



plt.legend(loc='lower right')
plt.ylabel("Fitness")
plt.xlabel("Gens")
plt.show()


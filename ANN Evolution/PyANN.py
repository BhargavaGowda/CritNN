import numpy as np
import random

class ANN:

    def __init__(self,size):
        self.size = size 
        self.weights = []
        self.bias = []

        for i in range(len(size)-1):
            self.weights.append(np.zeros((size[i+1],size[i])))
        
        for i in range(len(size)-1):
            self.bias.append(np.zeros(size[i+1]))
        

    @staticmethod
    def tanhAct(inputVector):
        top = np.exp(inputVector.clip(max=50,min=-50))-np.exp(-inputVector.clip(max=50,min=-50))
        bottom = np.exp(inputVector.clip(max=50,min=-50))+np.exp(-inputVector.clip(max=50,min=-50))
        return top/bottom
    
    def step(self,inputs):
        for i in range(len(self.size)-1):
            inputs = self.tanhAct(np.matmul(self.weights[i],inputs)+self.bias[i])
        return inputs
    
    def mutateSimple(self,mutationSize = 1):
        for i in range(len(self.size)-1):
            self.weights[i]+=np.random.normal(loc=0,scale=mutationSize,size=self.weights[i].shape)
            self.bias[i]+=np.random.normal(loc=0,scale=mutationSize,size=self.bias[i].shape)

    def mutateModular(self,mutationSize=1):
        layer = np.random.randint(1,len(self.size))
        neuron = np.random.randint(self.size[layer])
        self.weights[layer-1][neuron,:]+=np.random.normal(loc=0,scale=mutationSize,size=self.weights[layer-1][neuron,:].shape)
        self.bias[layer-1][neuron]+=np.random.normal(loc=0,scale=mutationSize)

    @staticmethod
    def copy(net):
        newNet = ANN(net.size)
        for i in range(len(net.size)-1):
            newNet.weights[i]=np.copy(net.weights[i])
            newNet.bias[i] = np.copy(net.bias[i])
        return newNet

    @staticmethod
    def getGeneDistance(net1,net2):
        output = 0
        for i in range(len(net1.size)-1):
            output += np.linalg.norm(net1.weights[i]-net2.weights[i])+np.linalg.norm(net1.bias[i]-net2.bias[i])
        return output
    
    @staticmethod
    def recombineModular(net1,net2):
        splitLayer = np.random.randint(len(net1.size)-1)
        splitPoint = np.random.randint(net1.size[splitLayer+1])
        newNet = ANN(net1.size)
        for layer in range(len(net1.size)-1):
            for neuron in range(net1.size[layer+1]):
                if layer>splitLayer or (layer == splitLayer and neuron>splitPoint):
                    newNet.weights[layer][neuron,:] = net2.weights[layer][neuron,:]
                    newNet.bias[layer][neuron] = net2.bias[layer][neuron]
                else:
                    newNet.weights[layer][neuron,:] = net1.weights[layer][neuron,:]
                    newNet.bias[layer][neuron] = net1.bias[layer][neuron]

        return newNet


            


        

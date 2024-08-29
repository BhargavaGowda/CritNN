import numpy
import random

class CTRNN:


    def __init__(self, size,weightRange=5,biasRange=5):
        self.size = size
        self.potentials = numpy.zeros(size)
        self.inputs = numpy.zeros(size)
        self.weights = (numpy.random.rand(size,size)-0.5)
        self.bias = (numpy.zeros(size))
        self.timescale = (numpy.full(size,0.5))
        self.weightRange = weightRange
        self.biasRange = biasRange
        self.reset()

    def getWeights(self):
        return self.weights

    def getBias(self):
        return self.bias

    def getTimescale(self):
        return self.timescale
    
    def getPotentials(self):
        return self.potentials
    
    def reset(self):
        self.potentials = numpy.zeros(self.size)
        self.inputs*=0

    def setInputs(self,maskVec):
        for i in range(self.size):
            self.weights[:,i] *= 1-maskVec

    def setOutputs(self,maskVec):
        for i in range(self.size):
            self.weights[i,:] *= 1-maskVec

    def step(self,inputArray):
        self.inputs+=inputArray
        self.inputs+=numpy.matmul(self.weights,self.potentials)
        self.inputs+=self.bias
        delta = (self.tanhAct(self.inputs)-self.potentials)
        self.potentials += self.timescale*delta
        self.inputs=numpy.zeros(self.size)
        return self.potentials


    def applyMask(self):
        self.weights = self.weights*self.mask



    @staticmethod
    def sigmoidAct(inputVector):
        bottom = 1+numpy.exp(-inputVector.clip(max=50,min=-50))
        return 1/bottom
    
    @staticmethod
    def tanhAct(inputVector):
        top = numpy.exp(inputVector.clip(max=50,min=-50))-numpy.exp(-inputVector.clip(max=50,min=-50))
        bottom = numpy.exp(inputVector.clip(max=50,min=-50))+numpy.exp(-inputVector.clip(max=50,min=-50))
        return top/bottom

    
    def lerpNet(self,otherNet,factor):
        self.weights+=factor*(otherNet.weights-self.weights)
        self.timescale+=factor*(otherNet.timescale-self.timescale)
        self.bias+=factor*(otherNet.bias-self.bias)
        self.weights = self.weights.clip(-1*self.weightRange,self.weightRange)
        self.bias = self.bias.clip(-1*self.biasRange,self.biasRange)
        self.timescale = self.timescale.clip(0.001,10)

    
    @staticmethod
    def getDistance(net1,net2):
        w = abs(numpy.linalg.norm(net1.weights-net2.weights))
        b = abs(numpy.linalg.norm(net1.bias-net2.bias))
        t = abs(numpy.linalg.norm(net1.timescale-net2.timescale))
        return w+b+t

    @staticmethod
    def recombine(brain1,brain2):

        #not comprehensive
        if(brain1.size != brain2.size or brain1.size<2):
            raise("brain mismatch")

        splitPoint = numpy.random.randint(1,brain1.size)
        newWeights = numpy.concatenate((brain1.weights[:splitPoint],brain2.weights[splitPoint:]))
        newBias = numpy.concatenate((brain1.bias[:splitPoint],brain2.bias[splitPoint:]))
        newTimescale = numpy.concatenate((brain1.timescale[:splitPoint],brain2.timescale[splitPoint:]))
        
        newBrain = CTRNN(brain1.size,brain1.weightRange,brain1.biasRange)
        newBrain.weights=newWeights
        newBrain.bias=newBias
        newBrain.timescale=newTimescale

        return newBrain
    
    @staticmethod
    def recombineModular(brain1,brain2):
        splitPoint = numpy.random.randint(brain1.size)

        newBrain = CTRNN(brain1.size)
        for i in range(newBrain.size):
            if i <splitPoint:
                newBrain.weights[i,:]=brain1.weights[i,:]
                newBrain.bias[i]=brain1.bias[i]
                newBrain.timescale[i]=brain1.timescale[i]
            else:
                newBrain.weights[i,:]=brain2.weights[i,:]
                newBrain.bias[i]=brain2.bias[i]
                newBrain.timescale[i]=brain2.timescale[i]

        return newBrain

    @staticmethod
    def copy(brain1):
        return CTRNN.recombine(brain1,brain1)
    
    def getGeneDistance(net1,net2):
        w = net1.weights-net2.weights
        b = net1.bias-net2.bias
        t = net1.timescale-net2.timescale
        return(numpy.count_nonzero(w)+numpy.count_nonzero(b)+numpy.count_nonzero(t))

    def mutateSimple(self, mutationSize = 1):
        self.weights = (self.weights+numpy.random.normal(loc=0,scale=mutationSize,size=self.weights.shape)).clip(-1*self.weightRange,self.weightRange)
        self.bias = (self.bias+numpy.random.normal(loc=0,scale=mutationSize,size=self.bias.shape)).clip(-1*self.biasRange,self.biasRange)
        self.timescale = (self.timescale+numpy.random.normal(loc=0,scale=0.1,size=self.timescale.shape)).clip(0.001,10)


    def mutatePointFull(self,mutationSize=1.0):
        w1 = numpy.random.randint(self.size)
        w2 = numpy.random.randint(self.size)
        b = numpy.random.randint(self.size)
        t = numpy.random.randint(self.size)
        self.weights[w1,w2]+=numpy.random.normal(loc=0,scale=mutationSize)
        self.bias[b]+=numpy.random.normal(loc=0,scale=mutationSize)
        # self.timescale[t] = (self.timescale[t]+numpy.random.randn()*0.1).clip(0,1)
        self.timescale[t] = numpy.clip((numpy.random.exponential(scale=1)),0,1)

    def mutateModular(self,mutationSize=1.0):
        neuron = numpy.random.randint(self.size)
        self.weights[neuron,:] = (self.weights[neuron,:]+ numpy.random.normal(loc=0,scale=mutationSize,size=self.weights[neuron,:].shape)).clip(-1*self.weightRange,self.weightRange)
        self.bias[neuron] = (self.bias[neuron]+numpy.random.normal()).clip(-1*self.biasRange,self.biasRange)
        self.timescale[neuron] = (self.timescale[neuron]+numpy.random.exponential()).clip(0,1)






        





        
        
        






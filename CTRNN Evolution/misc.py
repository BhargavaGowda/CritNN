import pickle
import numpy as np
from PyCTRNNv3 import CTRNN

with open("modelArchive\Lunar Lander\\150 gens\\best_net.pkl", "rb") as f:
    net1 = pickle.load(f)
    net1.reset()

with open("modelArchive\Lunar Lander\\20 gens\\best_net.pkl", "rb") as f:
    net2 = pickle.load(f)
    net2.reset()

# print(abs(net1.weights[8,:]-net2.weights[8,:]))
print(net1.weights[8,2],net1.weights[8,3])
print(net2.weights[8,2],net2.weights[8,3])
print(net1.timescale-net2.timescale)
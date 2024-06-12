from PyCTRNN import CTRNN
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

# env = gym.make("MountainCarContinuous-v0",render_mode="human")
env = gym.make("LunarLander-v2",continuous=True)

# net = CTRNN(14,weightRange=10,biasRange=10)
# net.mutateSimple(3)

# with open("net.pkl", "wb") as f:
#     pickle.dump(net, f)

with open("net.pkl", "rb") as f:
    net = pickle.load(f)
    net.reset()

imgSize = 25
paramStep = 0.001

i_off=0
j_off=0

T = 150

for t in range(T):
    with open("net.pkl", "rb") as f:
        net = pickle.load(f)
        net.reset()
    print("progress:",str(100*t/float(T))+"%")
    imgfit = np.zeros((imgSize,imgSize))
    img0 = np.zeros((imgSize,imgSize))
    img1 = np.zeros((imgSize,imgSize))
    img2 = np.zeros((imgSize,imgSize))
    a=0
    b=0

    for i in range(imgSize):     
        for j in range(imgSize):
            net.reset()
            observation, info = env.reset(seed=3)
            fitness = 0
            for _ in range(t*2+1):

                inp = np.array(observation)
                action = net.step(np.concatenate((inp,np.zeros(net.size-inp.size))))
                observation, reward, terminated, truncated, info = env.step(action[-2:])
                fitness+= reward

                if _==5:

                    net.potentials[11]+=(i-imgSize//2)*paramStep
                    net.potentials[30]+=(j-imgSize//2)*paramStep

                if terminated or truncated:
                    break

            # img[j,i] = observation[0]
            imgfit[j,i] = fitness
            img0[j,i] = observation[0]
            img1[j,i] = observation[1]
            img2[j,i] = observation[4]
            fitness = 0


    plt.imshow(imgfit,cmap="gnuplot",extent=[a+paramStep*(0-imgSize//2),a+paramStep*(imgSize-1-imgSize//2),b+paramStep*(0-imgSize//2),b+paramStep*(imgSize-1-imgSize//2)]
               ,origin = "lower",interpolation='none')#,vmin=-0.2,vmax=0.2)
    plt.colorbar()
    plt.savefig('ctrnn tests/figs/step_'+str(t)+'.png', bbox_inches='tight')
    plt.close()
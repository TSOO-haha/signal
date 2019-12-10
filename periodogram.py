import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math
pi=math.pi
#生成N个正弦信号
def generate_sin(N):
    '''

    :param N: number of sample
    :return: xn:the A series of signals
    '''
    # sample rate
    fs = 1000
    # N 个样本
    t = np.linspace(0, 0.3, N)
    # 振幅
    A = np.array([2, 8]).reshape(-1, 1)
    # 频率
    f = np.array([150, 140]).reshape(-1, 1)
    # 产生信号
    xn = (A * np.sin(2 * np.pi * f * t)).sum(axis=0) + 5 * np.random.randn(*t.shape)
    return xn
if __name__=="__main__":
    '''
    Step1:generate the a random signals
    '''
    # number of sample
    N = 301
    # 产生随机信号
    x = generate_sin(N)
    plt.plot(x)
    plt.xlabel("Frenquency")
    plt.ylabel("Amplitude")
    plt.show()
    '''
    Step2:Discrete Fourier Transform to x
    '''
    X = fft(x)
    # Magnitude(amplitude)
    mX = np.abs(X)
    # phase
    pX = np.angle(X)
    '''
    Step3:Calculate the |X(m)|^2/N
    '''
    P = (1/N)*np.square(mX)
    '''
    Step4: Plot the graph
    '''
    #归一化
    norm_P =[]
    max_P = np.max(P)
    min_P = np.min(P)
    for t in P:
        t = (t-min_P)/(max_P-min_P)
        norm_P.append(t)

    plt.plot(norm_P)
    plt.title("Periodogram")
    plt.xlabel("Frenquency")
    plt.ylabel("power spectrum")
    plt.show()


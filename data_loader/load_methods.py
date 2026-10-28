import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def CWRU(item_path):
    axis = ["_DE_time", "_FE_time", "_BA_time"]
    datanumber = os.path.basename(item_path).split(".")[0]

    if eval(datanumber) < 100:
        realaxis = "X0" + datanumber + axis[0]
    else:
        realaxis = "X" + datanumber + axis[0]
    signal = loadmat(item_path)[realaxis]

    return signal

def OU(item_path):
    axis = ["Channel_1", "Channel_2"]
    realaxis = axis[0]
    signal = loadmat(item_path)[realaxis]

    return signal

def MFPT(item_path):
    # f = item_path.split("/")[-2]
    print(item_path)
    if 'normal' in item_path:
        signal = (loadmat(item_path)["bearing"][0][0][1])
    else:
        signal = (loadmat(item_path)["bearing"][0][0][2])

    return signal


def PU(item_path):
    name = os.path.basename(item_path).split(".")[0]
    fl = loadmat(item_path)[name]
    signal = fl[0][0][2][0][6][2]  #Take out the data
    signal = signal.reshape(-1,1)

    return signal


def XJTU(item_path):
    fl = pd.read_csv(item_path)
    signal = fl["Horizontal_vibration_signals"]
    signal = signal.values.reshape(-1,1)

    return signal
    
def SEBEAR(item_path):
    axis = ["signal"]
    realaxis = axis[0]
    signal = loadmat(item_path)[realaxis][:,1]
    signal = signal.reshape(-1,1)

    return signal
    
def SEGEAR(item_path):
    axis = ["signal"]
    realaxis = axis[0]
    signal = loadmat(item_path)[realaxis][:,1]
    signal = signal.reshape(-1,1)

    return signal
    
def PU(item_path):
    axis = ["signal"]
    realaxis = axis[0]
    signal = loadmat(item_path)[realaxis]
    signal = signal.reshape(-1,1)
    return signal


def JNU(item_path):
    fl = pd.read_csv(item_path)
    # signal = fl["Horizontal_vibration_signals"]
    signal = fl.values.reshape(-1,1)

    return signal

# def IMS(item_path):
    # channel = {'normal': 0,
               # 'inner': 4,
               # 'outer': 0,
               # 'ball': 6}
    # f = item_path.split("/")[-2]
    # signal = np.loadtxt(item_path)[:, channel[f]]

    # return signal

def GB(item_path):
    signal = np.loadtxt(item_path)[: ,0]
    signal = signal.reshape(-1,1)
    return signal

def SQ(item_path):
    signal = np.loadtxt(item_path)[: ,1]
    signal = signal.reshape(-1,1)
    return signal


def GBN(item_path):
    signal = np.loadtxt(item_path)[:, 2]
    signal = signal.reshape(-1, 1)
    return signal


def GBR(item_path):
    signal = np.loadtxt(item_path)[:, 2]
    signal = signal.reshape(-1, 1)
    return signal

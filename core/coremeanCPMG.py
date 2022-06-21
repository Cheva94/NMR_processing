import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.optimize import curve_fit

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange', 'mediumseagreen', 'k', 'm', 'y'])

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5

plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def CPMG_file(File):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()

    t = data[:, 0] # In ms

    Re = data[:, 1]
    Im = data[:, 2]
    signal = Re + Im * 1j # Complex signal

    return t, signal

def PhCorr(signal):
    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    return signal * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))

def plot(t, signalArr, signalMean, nF, Out):
    fig, ax = plt.subplots()

    for k in range(nF):
        ax.plot(t, signalArr[:, k].real, ls=':', lw = 1)
    ax.plot(t, signalMean.real, label='mean')
    ax.set_xlabel('t [ms]')
    ax.set_ylabel('M')
    ax.legend()

    plt.savefig(f'{Out}')

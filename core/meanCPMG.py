import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
warnings.filterwarnings("ignore")

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

plt.rcParams["figure.figsize"] = 12.5, 20
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def CPMG_file(File, nini):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()

    Laplace = data[3:153, :3]
    T2 = Laplace[:, 0]
    Cumulative = Laplace[:, 2]

    Time = data[154:, :3]
    t = Time[:, 0]
    Decay = Time[:, 1]

    return t[nini:], Decay[nini:], T2, Cumulative

def PhCorr(signal, nH):
    RGnorm = 70
    norm = 1 / ((6.32589E-4 * np.exp(RGnorm/9) - 0.0854) * nH)

    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    return signal * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get))) * norm

def plot(t, signalArr, T2, cumulArr, nF, Out, Labels):
    fig, axs = plt.subplots(2,2)
    axins = inset_axes(axs[0,0], width="30%", height="30%", loc=5)

    for k in range(nF):
        axs[0,0].plot(t, signalArr[:, k], label=Labels[k])
        axins.plot(t[:30], signalArr[:30, k])

    axs[0,0].set_xlabel('t [ms]')
    axs[0,0].set_ylabel('CPMG')
    axs[0,0].legend()

    for k in range(nF):
        axs[0,1].plot(T2, cumulArr[:, k], label=Labels[k])

    axs[0,1].set_xlabel('T2 [ms]')
    axs[0,1].set_ylabel('Cumulative')
    axs[0,1].set_xscale('log')
    axs[0,1].legend()

    plt.savefig(f'{Out}')

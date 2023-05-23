import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
import scipy.fft as FT
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['coral', 'teal', 'tab:orange', 'mediumseagreen'])
plt.rcParams["axes.titlesize"] = "x-small"

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5

plt.rcParams["legend.loc"] = 'best'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 25, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linestyle"] = '-'

def FID_file(F, nini):
    data = pd.read_csv(F, header = None, delim_whitespace = True, comment='#').to_numpy()

    t = data[:, 0] # In ms

    Re = data[:, 1]
    Im = data[:, 2]
    signal = Re + Im * 1j

    return t[nini:], signal[nini:]

def PhCorrNorm(signal, nH):
    RGnorm = 70
    norm = 1 / ((6.32589E-4 * np.exp(RGnorm/9) - 0.0854) * nH)
    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    return signal * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get))) * norm

def plot(t, signalArr, nF, Out, Labels):
    fig, ax = plt.subplots()
    axins = inset_axes(ax, width="30%", height="30%", loc=5)

    for k in range(nF):
        ax.plot(t, signalArr[:, k].real, label=Labels[k])
        axins.plot(t[:30], signalArr[:30, k].real)

    ax.set_xlabel('t [ms]')
    ax.set_ylabel('FID')
    ax.legend()

    plt.savefig(f'{Out}')

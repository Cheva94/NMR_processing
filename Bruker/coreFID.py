import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
import scipy.fft as FT
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks
import warnings
import readbruker as rb
warnings.filterwarnings("ignore")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5

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

plt.rcParams["figure.figsize"] = 50, 20
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'

def FID_file(File):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    # read in the bruker formatted data
    dic, rawdata = rb.read(File)

    filter = 69
    sw = dic["acqus"]["SW_h"]

    sgl = rawdata[filter:]
    sgl[0] = 2*sgl[0] # arreglo lo que el bruker rompe
    nP = len(sgl)
    t = np.array([x * 1E6 / sw for x in range(nP)]) # en microsegundos

    return t, sgl, nP, sw


def PhCorr(sgl):
    '''
    Corrección de fase.
    '''

    maxVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        sgl_ph = sgl * np.exp(1j * tita)
        maxVal[i] = np.max(sgl_ph.real[0:30])
    sgl *= np.exp(1j * np.deg2rad(max(maxVal, key=maxVal.get)))
    
    return sgl


def espec(sgl, nP, sw):
    '''
    Creación del espectro.
    '''

    # Preparación del espectro
    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=1/sw)) # Hz scale
    CS = freq / 300 # ppm for Bruker scale
    spec = np.flip(FT.fftshift(FT.fft(sgl, n = zf)))

    return CS, spec
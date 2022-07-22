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

def FID_file(File, nini):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    data = pd.read_csv(File, header = None, sep='\t').to_numpy()

    t = data[:, 0] # In ms
    DW = t[1] - t[0] # Dwell time
    nP = len(t) - nini # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    signal = Re + Im * 1j

    pAcq = pd.read_csv(File.split(".txt")[0]+'_acqs.txt', header = None,  sep='\t')
    nS, RDT, RG, att, RD, p90 = pAcq.iloc[0, 1], pAcq.iloc[1, 1], pAcq.iloc[2, 1], pAcq.iloc[3, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1]

    return t[nini:], signal[nini:], nP, DW, nS, RDT, RG, att, RD, p90

def PhCorr(signal):
    '''
    Corrección de fase.
    '''

    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    return signal * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))

def Norm(signal, RG):
    '''
    Normalización por ganancia.
    '''

    norm = 1 / (6.32589E-4 * np.exp(RG/9) - 0.0854)
    return signal * norm

def plot(t, signal, nP, DW, nS, RDT, RG, att, RD, p90, Out, Back, nini):
    '''
    Grafica resultados.
    '''

    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(rf'nS={nS:.0f}    |    RDT = {RDT} ms    |    RG = {RG:.0f} dB    |    Atten = {att:.0f} dB    |    RD = {RD:.0f} s    |    p90 = {p90} $\mu$s', fontsize='large')

    # Promedio de los primeros 10 puntos de la FID
    points = 10
    fid0Arr = signal[0:points].real
    fid0 = sum(fid0Arr) / points
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

    # Plot de la parte real de la FID
    axs[0,0].set_title(f'Se descartaron {nini:.0f} puntos al comienzo.', fontsize='large')
    axs[0,0].plot(t, signal.real, label='FID (real)')
    axs[0,0].plot(t[0:points], signal[0:points].real, lw = 10, label = fr'$M_R ({points})$ = ({fid0:.2f} $\pm$ {fid0_SD:.2f})')
    axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,0].set_xlabel('t [ms]')
    axs[0,0].set_ylabel('FID')
    axs[0,0].legend()

    # Inset del comienzo de la parte real de la FID
    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(t[0:40], signal[0:40].real, color='coral')
    axins1.plot(t[0:points], signal[0:points].real, color='teal')

    # Plot de la parte imaginaria de la FID
    axs[1,0].plot(t, signal.imag, label='FID (imag)')
    axs[1,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,0].set_xlabel('t [ms]')
    axs[1,0].set_ylabel('FID')
    axs[1,0].legend()

    # Preparación del espectro
    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
    CS = freq / 20 # ppm for Minispec scale
    spec = np.flip(FT.fftshift(FT.fft(signal, n = zf)))
    mask = (CS>-0.05)&(CS<0.05)
    max_peak = np.max(spec.real[mask])
    spec /= max_peak
    area_peak = np.sum(spec.real[mask])

    # Plot de la parte real del espectro, zoom en el pico
    axs[0,1].set_title(f'¿Background restado? {Back}', fontsize='large')
    axs[0,1].plot(CS, spec.real, label='Spectrum (real)')
    axs[0,1].fill_between(CS[mask], 0, spec.real[mask], label = fr'Peak area = {area_peak:.0f}', alpha = 0.25, color="teal")
    axs[0,1].set_xlim(-0.1, 0.1)
    axs[0,1].set_ylim(-0.05, 1.2)
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,1].set_xlabel(r'$\delta$ [ppm]')
    axs[0,1].axvline(x=0, color='k', ls=':', lw=2)
    axs[0,1].axhline(y=0, color='k', ls=':', lw=2)
    axs[0,1].legend()

    # Inset del espectro completo
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
    axins2.tick_params(labelleft=False)
    axins2.plot(CS, spec.real)

    # Plot de la parte imaginaria del espectro
    axs[1,1].plot(CS, spec.imag, label='Spectrum (imag)')
    axs[1,1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1,1].axvline(x=0, color='k', ls=':', lw=4)
    axs[1,1].set_xlim(-0.1, 0.1)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].legend()

    plt.savefig(f'{Out}')

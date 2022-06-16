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

plt.rcParams["figure.figsize"] = 50, 20
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linestyle"] = '-'

def FID_file(File, nini):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()

    t = data[:, 0] # In ms
    DW = t[1] - t[0] # Dwell time
    nP = len(t) - nini # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    signal = Re + Im * 1j

    pAcq = pd.read_csv(File.split(".txt")[0]+'-acqs.txt', header = None, delim_whitespace = True)
    nS, RG, p90, att, RD = pAcq.iloc[0, 1], pAcq.iloc[1, 1], pAcq.iloc[2, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1]

    return t[nini:], signal[nini:], nP, DW, nS, RG, p90, att, RD

def PhCorr(signal):
    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    return signal * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))

def Norm(signal, RGnorm, RG, m):
    if RGnorm == "off":
        Norm = 1 / m
    elif RGnorm == 'on':
        Norm = 1 / ((6.32589E-4 * np.exp(RG/9) - 0.0854) * m)
    return signal * Norm

def plot(t, signal, nP, DW, nS, RGnorm, RG, p90, att, RD, Out, Back, m, nini):
    points = 10
    fid0Arr = signal[0:points].real
    fid0 = sum(fid0Arr) / points
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / points) ** 0.5

    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
    CS = freq / 20 # ppm for Minispec scale
    spec = np.flip(FT.fftshift(FT.fft(signal, n = zf)))
    max_peak = np.max(spec)
    area_peak = np.sum(spec.real[260887:263405])
    spec /= max_peak

    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3,1]})

    fig.suptitle(f'nS={nS} | RG = {RG} dB ({RGnorm}) | RD = {RD} s | p90 = {p90} us | Atten = {att} dB | BG = {Back} | m = {m} | nini = {nini}', fontsize='small')

    axs[0,0].plot(t, signal.real, label='FID (real)')
    axs[0,0].plot(t[0:points], signal[0:points].real, lw = 10, label = fr'$M_R ({points})$ = ({fid0:.2f} $\pm$ {fid0_SD:.2f})')
    axs[0,0].axhline(y=0, color='teal', ls=':', lw=4)
    axs[0,0].set_xlabel('t [ms]')
    axs[0,0].set_ylabel('M')
    axs[0,0].legend()

    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.plot(t[0:30], signal[0:30].real)
    axins1.plot(t[0:points], signal[0:points].real, lw = 10)

    axs[1,0].plot(t, signal.imag, label='FID (imag)')
    axs[1,0].axhline(y=0, color='teal', ls=':', lw=4)
    axs[1,0].set_xlabel('t [ms]')
    axs[1,0].set_ylabel('M')
    axs[1,0].legend()

    axs[0,1].plot(CS, spec.real, label='Spectrum (real)')
    axs[0,1].fill_between(CS[260887:263405], 0, spec.real[260887:263405], label = fr'Peak area = {area_peak*1e-6:.2f}x10$^6$')
    axs[0,1].set_xlim(-0.1, 0.1)
    axs[0,1].set_ylim(-0.05, 1.2)
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,1].set_xlabel(r'$\delta$ [ppm]')
    axs[0,1].axvline(x=0, color='teal', ls=':', lw=4)
    axs[0,1].axhline(y=0, color='teal', ls=':', lw=4)
    axs[0,1].legend()

    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=2)
    axins2.tick_params(labelleft=False)
    axins2.plot(CS, spec.real)

    axs[1,1].plot(CS, spec.imag, label='Spectrum (imag)')
    axs[1,1].axhline(y=0, color='teal', ls=':', lw=4)
    axs[1,1].axvline(x=0, color='teal', ls=':', lw=4)
    axs[1,1].set_xlim(-0.1, 0.1)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_xlabel(r'$\delta$ [ppm]')
    axs[1,1].legend()

    plt.savefig(f'{Out}')

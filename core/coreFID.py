#!/usr/bin/python3.6
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
import scipy.fft as FT

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange', 'mediumseagreen', 'm', 'y', 'k'])
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

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linestyle"] = '-'

def FID_file(File):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()

    t = data[:, 0] # In ms
    DW = t[1] - t[0] # Dwell time
    nP = len(t) # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    signal = Re + Im * 1j

    pAcq = pd.read_csv(File.split(".txt")[0]+'-acqs.txt', header = None, delim_whitespace = True)
    nS, RG, p90, att, RD = pAcq.iloc[0, 1], pAcq.iloc[1, 1], pAcq.iloc[2, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1]

    return t, signal, nP, DW, nS, RG, p90, att, RD

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

def plot(t, signal, nP, DW, nS, RGnorm, RG, p90, att, RD, Out, Back, m):
    fid0Arr = signal[0:5].real
    fid0 = sum(fid0Arr) / 5
    fid0_SD = (sum([((x - fid0) ** 2) for x in fid0Arr]) / 5) ** 0.5

    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
    spec = FT.fftshift(FT.fft(signal, n = zf))
    max_peak = np.max(spec)
    spec /= max_peak
    CS = freq / 20

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(f'nS={nS} | RG = {RG} dB ({RGnorm}) | RD = {RD} s | p90 = {p90} us | Atten = {att} dB | BG = {Back} | m = {m}', fontsize='small')

    ax1.plot(t, signal.real)
    ax1.set_xlabel('t [ms]')
    ax1.set_ylabel('M')
    ax1.text(0.98,0.98, fr'$M_R (0)$ = ({fid0:.2f} $\pm$ {fid0_SD:.2f})', ha='right', va='top', transform=ax1.transAxes, fontsize='small')

    ax2.plot(CS, spec.real)
    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-0.2, 1.2)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_xlabel(r'$\delta$ [ppm]')
    ax2.axvline(x=0, color='gray', ls=':', lw=2)
    ax2.text(0.98,0.98, f'Peak = {max_peak.real:.2f}', ha='right', va='top', transform=ax2.transAxes, fontsize='small')

    plt.savefig(f'{Out}')

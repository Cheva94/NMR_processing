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

plt.rcParams["figure.figsize"] = 12.5, 13.5
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

def PhCorrNorm(signal, RGnorm, RG, m):
    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    if RGnorm == "off":
        Norm = 1 / m
    elif RGnorm == 'on':
        Norm = 1 / ((6.32589E-4 * np.exp(RG/9) - 0.0854) * m)

    return signal * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get))) * Norm

def FID(t, signal, nS, RGnorm, RG, p90, att, RD, Out):
    fid0ReArr = signal[0:5].real
    fid0Re = sum(fid0ReArr) / 5
    fid0Re_SD = (sum([((x - fid0Re) ** 2) for x in fid0ReArr]) / 5) ** 0.5

    fid0ImArr = signal[0:5].imag
    fid0Im = sum(fid0ImArr) / 5
    fid0Im_SD = (sum([((x - fid0Im) ** 2) for x in fid0ImArr]) / 5) ** 0.5

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax1.set_title(f'nS={nS} ; RG = {RG} dB ({RGnorm}) ; RD = {RD} s \n p90 = {p90} us ; Atten = {att} dB')

    ax1.plot(t, signal.real)
    ax1.set_xlabel('t [ms]')
    ax1.set_ylabel(r'$M_R$')
    ax1.text(0.98,0.98, fr'$M_R (0; 5)$ = ({fid0Re:.2f} $\pm$ {fid0Re_SD:.2f})', ha='right', va='top', transform=ax1.transAxes)

    ax2.plot(t, signal.imag, color='mediumseagreen')
    ax2.xaxis.tick_top()
    ax2.set_ylabel(r'$M_I$')
    ax2.text(0.98,0.02, fr'$M_I (0; 5)$ = ({fid0Im:.2f} $\pm$ {fid0Im_SD:.2f})', ha='right', va='bottom', transform=ax2.transAxes)

    plt.savefig(f'{Out}-DomTemp')

    with open(f'{Out}-DomTemp.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s] \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD} \n\n')

        f.write("t [ms], Re[FID], Im[FID] \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.6f}, {signal.real[i]:.6f}, {signal.imag[i]:.6f} \n')

def spectrum(signal, nP, DW, nS, RGnorm, RG, p90, att, RD, Out):
    zf = FT.next_fast_len(2**5 * nP)
    freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
    spec = FT.fftshift(FT.fft(signal, n = zf))
    max_peak = np.max(spec)

    specNorm = spec / max_peak
    CS = freq / 20

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    axs[0].set_title(f'nS={nS} ; RG = {RG} dB ({RGnorm}) ; RD = {RD} s \n p90 = {p90} us ; Atten = {att} dB')

    axs[0].plot(CS, specNorm.real)
    axs[0].set_xlim(-0.1, 0.1)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].set_xlabel(r'$\delta$ [ppm]')
    axs[0].set_ylabel(r'$M_R$')
    axs[0].axvline(x=0, color='gray', ls=':', lw=2)
    axs[0].text(0.98,0.98, f'Peak = {max_peak.real:.2f}', ha='right', va='top', transform=axs[0].transAxes, fontsize='small')

    axs[1].plot(CS, specNorm.imag, color='mediumseagreen')
    axs[1].set_xlim(-0.1, 0.1)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1].set_ylabel(r'$M_I$')
    axs[1].xaxis.tick_top()

    plt.savefig(f'{Out}-DomFreq')

    with open(f'{Out}-DomFreq.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s] \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD} \n\n')

        f.write("Freq [Hz], CS [ppm], Re[spec], Im[spec] \n")
        for i in range(len(freq)):
            f.write(f'{freq[i]:.6f}, {CS[i]:.6f}, {spec.real[i]:.6f}, {spec.imag[i]:.6f} \n')

# def back_subs(FID, back):
#     '''
#     Substract the given background to the FID.
#     '''
#
#     Re = FID.real - back.real
#     Im = FID.imag - back.imag
#
#     return Re + Im * 1j

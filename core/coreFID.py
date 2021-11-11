#!/usr/bin/python3.6

'''
    Description: core functions for FID.py.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
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
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange',
                                        'mediumseagreen', 'm', 'y', 'k'])

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

def userfile(F):
    '''
    Extracts data from the .txt input file given by the user.
    '''

    data = pd.read_csv(F, header = None, delim_whitespace = True).to_numpy()

    t = data[:, 0] # In ms
    DW = t[1] - t[0] # Dwell time
    # SW = 1000 / DW # Spectral width in Hz
    nP = len(t) # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    FID = Re + Im * 1j # Complex signal

    acq = F.split('.txt')[0]+'-acqs'+'.txt'
    acq = pd.read_csv(acq, header = None, delim_whitespace = True)
    nS, RG, RD = acq.iloc[0, 1], acq.iloc[1, 1], acq.iloc[5, 1]

    return t, nP, DW, FID, nS, RD, RG

def phase_correction(FID):
    '''
    Returns FID with phase correction (maximizing real part).
    '''

    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        FID_ph = FID * np.exp(1j * tita)
        initVal[i] = FID_ph[0].real

    return FID * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))

def normalize(FID, nH, RG):
    '''
    Normalizes FID considering the receiver gain and the number of protons.
    '''

    return 100 * FID / (RG * nH)

def plot_FID(t, FID, nS, RD, fileRoot):
    '''
    Plots normalized FID (real and imaginary parts).
    '''

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(t, FID.real)
    ax1.set_xlabel('t [ms]')
    ax1.set_ylabel(r'$M_R$')
    ax1.set_title(f'nS={nS} ; RD = {RD}')
    ax1.text(0.98,0.98, fr'$M_R (0)$ = {FID[0].real:.2f}', ha='right',
            va='top', transform=ax1.transAxes)

    ax2.plot(t, FID.imag, label='Im', color='mediumseagreen')
    ax2.xaxis.tick_top()
    ax2.set_ylabel(r'$M_I$')
    ax2.text(0.98,0.02, fr'$M_I (0)$ = {FID[0].imag:.2f}', ha='right',
            va='bottom', transform=ax2.transAxes)

    plt.savefig(f'{fileRoot}_NormPhCorr')

def out_FID(t, FID, fileRoot):
    '''
    Generates output file with normalized and phase corrected FID.
    '''

    with open(f'{fileRoot}_NormPhCorr.csv', 'w') as f:
        f.write("t [ms], Re[FID], Im[FID] \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {FID.real[i]:.4f}, {FID.imag[i]:.4f} \n')

def spectrum(FID, nP, DW):
    '''
    Creates normalized spectrum from FID signal and its frequency axis.
    '''

    freq = FT.fftshift(FT.fftfreq(nP, d=DW)) # Hz scale
    spec = FT.fftshift(FT.fft(FID))
    max_peak = np.max(spec)

    return freq, spec, max_peak

def plot_spec(freq, spec, max_peak, nS, RD, fileRoot):
    '''
    Plots spectrum (real and imaginary parts) in Hz and ppm (considering 20 MHz for MiniSpec).
    '''

    spec /= max_peak

    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]}, figsize= (25, 13.5))

    axs[0,0].plot(freq, spec.real)
    axs[0,0].set_xlim(-2, 2)
    axs[0,0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,0].set_xlabel(r'$\nu$ [Hz]')
    axs[0,0].set_ylabel(r'$M_R$')

    axs[1,0].plot(freq, spec.imag, color='mediumseagreen')
    axs[1,0].set_xlim(-2, 2)
    axs[1,0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,0].set_ylabel(r'$M_I$')
    axs[1,0].xaxis.tick_top()

    CS = freq / 20

    axs[0,1].plot(CS, spec.real)
    axs[0,1].set_xlim(-0.2, 0.2)
    axs[0,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0,1].set_xlabel(r'$\delta$ [ppm]')
    axs[0,1].set_ylabel(r'$M_R$')

    axs[1,1].plot(CS, spec.imag, color='mediumseagreen')
    axs[1,1].set_xlim(-0.2, 0.2)
    axs[1,1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1,1].set_ylabel(r'$M_I$')
    axs[1,1].xaxis.tick_top()

    fig.suptitle(f'nS={nS} ; RD = {RD} ; Peak = {max_peak.real:.2f}')
    plt.savefig(f'{fileRoot}_spectrum')

def out_spec(freq, spec, fileRoot):
    '''
    Generates output file with the spectrum.
    '''

    CS = freq / 20
    with open(f'{fileRoot}_spectrum.csv', 'w') as f:
        f.write("Freq [Hz], CS [ppm], Re[spec], Im[spec] \n")
        for i in range(len(freq)):
            f.write(f'{freq[i]:.4f}, {CS[i]:.4f}, {spec.real[i]:.4f}, {spec.imag[i]:.4f} \n')

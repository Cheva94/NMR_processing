#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
import numpy as np
import pandas as pd
import scipy.fft as FT
import matplotlib.pyplot as plt

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

plt.rcParams["figure.figsize"] = 37.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'

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

def main():

    FileArr = args.input
    Out = args.output
    Labels = args.labels

    nF = range(len(FileArr))

    fig, axs = plt.subplots(1,3)

    for k in nF:
        data = pd.read_csv(FileArr[k], header = None, delim_whitespace = True).to_numpy()
        t = data[:, 0] # In ms
        DW = t[1] - t[0]
        nP = len(t) # Number of points

        Re = data[:, 1]
        Im = data[:, 2]
        signal = Re + Im * 1j
        signal = PhCorr(signal)

        axs[0].plot(t, signal.real, label=Labels[k])
        axs[1].plot(t, signal.real/np.max(signal.real), label=Labels[k])

        # Preparación del espectro
        zf = FT.next_fast_len(2**5 * nP)
        freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
        CS = freq / 20 # ppm for Minispec scale
        spec = np.flip(FT.fftshift(FT.fft(signal, n = zf)))
        mask = (CS>-0.05)&(CS<0.05)
        if k==0:
            max_peak = np.max(spec.real[mask])
        spec /= max_peak

        axs[2].plot(CS, spec.real, label=Labels[k])

    axs[0].set_xlabel('t [ms]')
    axs[0].set_ylabel('FID')
    axs[0].set_xlim(right=20)
    axs[0].legend()

    axs[1].set_xlabel('t [ms]')
    axs[1].set_ylabel('FID (norm)')
    axs[1].set_xlim(right=20)
    axs[1].legend()

    axs[2].set_xlim(-0.06, 0.06)
    axs[2].set_ylim(-0.05, 1.2)
    axs[2].set_xlabel(r'$\delta$ [ppm]')
    axs[2].axvline(x=0, color='k', ls=':', lw=2)
    axs[2].axhline(y=0, color='k', ls=':', lw=2)
    axs[2].legend()

    plt.savefig(f'{Out}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-L', '--labels', nargs='+')

    args = parser.parse_args()

    main()

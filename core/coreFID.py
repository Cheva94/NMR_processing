#!/usr/bin/python3.8

'''
    Description: core functions for FID.py.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.fft as FT

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 3
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange',
                                        'mediumseagreen', 'm', 'y', 'k'])

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 3

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

def userfile(input_file):
    '''
    Process the txt input file given by the user.
    '''

    data = pd.read_csv(input_file, header = None, delim_whitespace = True).to_numpy()

    t = data[:, 0]
    Np = len(t) # Number of points

    dw = t[1] - t[0] # Dwell time (t in ms)
    # sw = 1000 / dw # Spectral width in Hz (dw in ms)

    Re = data[:, 1]
    Im = data[:, 2]
    FID = Re + Im * 1j # Complex signal

    acq = input_file.split('.txt')[0]+'_acqs'+'.txt'
    data_acq = pd.read_csv(acq, header = None, delim_whitespace = True)
    ns, rg, rd = data_acq.iloc[0, 1], data_acq.iloc[1, 1], data_acq.iloc[5, 1]

    return t, Np, dw, FID, ns, rd, rg

def phase_correction(FID):
    '''
    Returns FID with phase correction (maximize real part).
    '''

    initVal = {}

    for i in range(360):
        tita = np.deg2rad(i)
        FID_ph = FID * np.exp(1j * tita)
        initVal[i] = FID_ph[0].real

    return FID * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))

def spectrum(FID, Np, dw):
    '''
    Creates normalized spectrum from FID signal and its frequency axis.
    '''

    freq_axis = FT.fftshift(FT.fftfreq(Np, d=dw)) # Hz scale

    spec = FT.fftshift(FT.fft(FID))
    max_peak = np.max(spec)
    spec /= max_peak

    return freq_axis, spec, max_peak

def plot_FID(t, FID, ns, rd, rg, input_file):

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(t, FID.real, label='Re')
    ax1.set_xlabel('t [ms]')
    ax1.set_ylabel(r'$\Re$[s(t)]')
    ax1.set_title(f'ns={ns} ; rd = {rd} ; rg = {rg}')
    ax1.text(0.98,0.98, fr'$\Re$[s(0)] = {FID[0].real:.2f}', ha='right',
            va='top', transform=ax1.transAxes)

    ax2.plot(t, FID.imag, label='Im', color='mediumseagreen')
    ax2.xaxis.tick_top()
    ax2.set_ylabel(r'$\Im$[s(t)]')
    ax2.text(0.98,0.02, fr'$\Im$[s(0)] = {FID[0].imag:.2f}', ha='right',
            va='bottom', transform=ax2.transAxes)

    plt.savefig(f'{input_file.split(".txt")[0]}-FID')

def plot_spec_freq(freq, spec, max_peak, ns, rd, rg, input_file):

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(freq, spec.real)
    ax1.set_xlabel(r'$\nu$ [Hz]')
    ax1.set_ylabel(r'$\Re$[S($\nu$)]')
    ax1.set_title(f'ns={ns} ; rd = {rd} ; rg = {rg}')
    ax1.text(0.98,0.98, f'Max = {max_peak.real:.2f}', ha='right',
            va='top', transform=ax1.transAxes)

    ax2.plot(freq, spec.imag, color='mediumseagreen')
    ax2.set_ylabel(r'$\Im$[S($\nu$)]')
    ax2.xaxis.tick_top()

    plt.savefig(f'{input_file.split(".txt")[0]}-spec_freq')

def plot_spec_mini(freq, spec, max_peak, ns, rd, rg, input_file):

    CS = freq / 20

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(CS, spec.real)
    ax1.set_xlabel(r'$\delta$ [ppm]')
    ax1.set_ylabel(r'$\Re$[S($\nu$)]')
    ax1.set_title(f'ns={ns} ; rd = {rd} ; rg = {rg}')
    ax1.text(0.98,0.98, f'Max = {max_peak.real:.2f}', ha='right',
            va='top', transform=ax1.transAxes)

    ax2.plot(CS, spec.imag, color='mediumseagreen')
    ax2.set_ylabel(r'$\Im$[S($\nu$)]')
    ax2.xaxis.tick_top()

    plt.savefig(f'{input_file.split(".txt")[0]}-spec_MiniSpec')

def plot_spec_bruker(freq, spec, max_peak, ns, rd, rg, input_file):

    CS = freq / 300

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(CS, spec.real)
    ax1.set_xlabel(r'$\delta$ [ppm]')
    ax1.set_ylabel(r'$\Re$[S($\nu$)]')
    ax1.set_title(f'ns={ns} ; rd = {rd} ; rg = {rg}')
    ax1.text(0.98,0.98, f'Max = {max_peak.real:.2f}', ha='right',
            va='top', transform=ax1.transAxes)

    ax2.plot(CS, spec.imag, color='mediumseagreen')
    ax2.set_ylabel(r'$\Im$[S($\nu$)]')
    ax2.xaxis.tick_top()

    plt.savefig(f'{input_file.split(".txt")[0]}-spec_Bruker')

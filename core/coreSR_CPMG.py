#!/usr/bin/python3.6

'''
    Description: core functions for SR_CPMG.py.

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
# import scipy.fft as FT

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange', 'mediumseagreen', 'm', 'y', 'k'])

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

def flint(K1, K2, Z, alpha, S):
    '''
    Fast 2D NMR relaxation distribution estimation.
    '''

    iter_max = 100000

    K1TK1 = K1.T @ K1
    K2TK2 = K2.T @ K2
    K1TZK2 = K1.T @ Z @ K2
    resZZT = np.trace(Z @ Z.T) # used for calculating residual

    L = 2 * (np.trace(K1TK1) * np.trace(K2TK2) + alpha) # Lipschitz constant is larger than largest eigenvalue, but not much larger and with rapid decay. The factor of 2 helps
    # Ver de simplificar los dos próximos renglones
    fac1 = (L - 2 * alpha) / L
    fac2 = 2 / L
    lastRes = np.inf

    resida = np.full((iter_max, 1), np.nan) # Vector columna

    # Todo lo anterior es preparativo, recién ahora arranca la el algoritmo FISTA
    Y = S
    tstep = 1

    for iter in range(iter_max):
        term2 = K1TZK2 - K1TK1 @ Y @ K2TK2
        Snew = fac1 * Y + fac2 * term2
        Snew = np.maximum(0, Snew)

        tnew = 0.5 * (1 + np.sqrt(1 + 4 * tstep**2))
        fac3 = (tstep - 1) / tnew
        Y = Snew + fac3 * (Snew - S)
        tstep = tnew
        S = Snew

        # Don't calculate the residual every iteration; it takes much longer than the rest of the algorithm
        if iter % 500 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            resid = resZZT - 2 * np.trace(S.T @ K1TZK2) + np.trace(S.T @ K1TK1 @ S @ K2TK2) + TikhTerm
            resida[iter] = resid
            resd = np.abs(resid - lastRes) / resid
            lastRes = resid
            # Show progress
            print(f'iter = {iter} ; tstep = {tstep} ; trat = {fac3} ; L = {L} ; resid = {resid} ; resd = {resd}')

            if resd < 1E-5: # truncation threshold
                break

    return S, resida

def plot_map(T2, T1, S):
    '''
    hkjh
    '''

    fig, ax1 = plt.subplots()

    ax1.contour(T2, T1, S, 1000)
    ax1.set_xlabel(r'$T_2$ [ms]')
    ax1.set_ylabel(r'$T_1$ [ms]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    plt.savefig(f'RatesSpectrum')
    # Ver qué onda el tema de las diagonales cocientes de T1/T2

def plot_proj(T2, T1, S):
    '''
    sdasd
    '''

    projT1 = np.sum(S, axis=1)
    projT2 = np.sum(S, axis=0)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25, 10))

    ax1.plot(T1, projT1)
    ax1.set_xlabel(r'$T_1$ [ms]')
    ax1.set_xscale('log')

    ax2.plot(T2, projT2)
    ax2.set_xlabel(r'$T_2$ [ms]')
    ax2.set_xscale('log')

    plt.savefig(f'RatesSpectrum_proj')

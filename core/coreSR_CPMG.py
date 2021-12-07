#!/usr/bin/python3.6

'''
    Description: core functions for SR_CPMG.py.

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib import cm
from matplotlib.ticker import LinearLocator

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

def userfile(File, fileRoot, Nx, Ny, T1min, T1max, T2min, T2max, niniT1, niniT2):
    '''
    Extracts data from the .txt input file given by the user.
    '''

    S0 = np.ones((Nx, Ny))
    T1 = np.logspace(T1min, T1max, Nx)
    T2 = np.logspace(T2min, T2max, Ny)

    tau1 = pd.read_csv(f'{fileRoot+"_t1.dat"}', header = None, delim_whitespace = True).to_numpy()
    tau2 = pd.read_csv(f'{fileRoot+"_t2.dat"}', header = None, delim_whitespace = True).to_numpy()
    N1, N2 = len(tau1), len(tau2)
    tau1 = tau1[niniT1:]
    tau2 = tau2[niniT2:]

    K1 = 1 - np.exp(-tau1 / T1)
    K2 = np.exp(-tau2 / T2)

    data = pd.read_csv(File, header = None, delim_whitespace = True).to_numpy()[:, 0]
    # Re = data[:, 0]
    # Im = data[:, 1]
    # decay = Re + Im * 1j # Complex signal
    # Por acá haría falta la corrección de fase

    Z = np.reshape(data, (N1, N2))[niniT2:, niniT1:]

    return S0, T1, T2, tau1, tau2, K1, K2, Z

def plot_Z(tau1, tau2, Z, fileRoot):
    '''
    adasda
    '''

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25, 10))

    ax1.plot(tau1, Z[:, 0])
    ax1.set_xlabel(r'$\tau_1$ [ms]')
    ax1.set_ylabel('SR')

    ax2.plot(tau2, Z[-1, :])
    ax2.set_xlabel(r'$\tau_2$ [ms]')
    ax2.set_ylabel('CPMG')

    plt.savefig(f'{fileRoot}-PhCorrZ')

def NLI_FISTA(K1, K2, Z, alpha, S):
    '''
    Fast 2D NMR relaxation distribution estimation.
    '''

    K1TK1 = K1.T @ K1
    K2TK2 = K2.T @ K2
    K1TZK2 = K1.T @ Z @ K2

    L = 2 * (np.trace(K1TK1) * np.trace(K2TK2) + alpha) # Lipschitz constant is larger than largest eigenvalue, but not much larger and with rapid decay. The factor of 2 helps

    resZZT = np.trace(Z @ Z.T) # used for calculating residual

    # Ver de simplificar los dos próximos renglones
    fac1 = (L - 2 * alpha) / L
    fac2 = 2 / L
    lastRes = np.inf

    # Todo lo anterior es preparativo, recién ahora arranca la el algoritmo FISTA
    Y = S
    tstep = 1
    iter_max = 100000
    resida = np.full((iter_max, 1), np.nan) # Vector columna

    for iter in range(iter_max):
        term2 = K1TZK2 - K1TK1 @ Y @ K2TK2
        Snew = fac1 * Y + fac2 * term2
        # Snew = np.maximum(0, Snew)
        Snew[Snew<0] = 0

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

def plot_map(T1, T2, S, nLevel, fileRoot):
    '''
    hkjh
    '''

    fig, ax = plt.subplots()

    ax.contour(T2, T1, S, nLevel)
    ax.set_xlabel(r'$T_2$ [ms]')
    ax.set_ylabel(r'$T_1$ [ms]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(f'{fileRoot}_CharTimeSpectrum')
    # Ver qué onda el tema de las diagonales cocientes de T1/T2

def plot_proj(T1, T2, S, fileRoot):
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

    plt.savefig(f'{fileRoot}_Projections')

#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from scipy.signal import find_peaks
import matplotlib as mpl

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

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linestyle"] = '-'

def SRCPMG_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()#[:, 0]
    Re = data[:, 0]
    Im = data[:, 1]
    signal = Re + Im * 1j # Complex signal

    Nx = Ny = 150
    S0 = np.ones((Nx, Ny))
    T1 = np.logspace(T1min, T1max, Nx)
    T2 = np.logspace(T2min, T2max, Ny)

    tau1 = pd.read_csv(File.split('.txt')[0]+"_t1.dat", header = None, delim_whitespace = True).to_numpy()
    tau2 = pd.read_csv(File.split('.txt')[0]+"_t2.dat", header = None, delim_whitespace = True).to_numpy()
    N1, N2 = len(tau1), len(tau2)
    tau1 = tau1[niniT1:]
    tau2 = tau2[niniT2:]

    K1 = 1 - np.exp(-tau1 / T1)
    K2 = np.exp(-tau2 / T2)

    return S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2

def PhCorr(signal, N1, N2, niniT1, niniT2):
    Z = []

    for k in range(N1):
        initVal = {}
        signal_k = signal[k*N2:(k+1)*N2]
        for i in range(360):
            tita = np.deg2rad(i)
            signal_ph = signal_k * np.exp(1j * tita)
            initVal[i] = signal_ph[0].real

        signal_k = signal_k * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))
        Z.append(signal_k.real)

    return np.reshape(Z, (N1, N2))[niniT1:, niniT2:]

def Norm(Z, RGnorm, m):
    if RGnorm == None:
        Norm = 1 / m
    else:
        Norm = 1 / ((6.32589E-4 * np.exp(RGnorm/9) - 0.0854) * m)
    return Z * Norm

def NLI_FISTA(K1, K2, Z, alpha, S):
    K1TK1 = K1.T @ K1
    K2TK2 = K2.T @ K2
    K1TZK2 = K1.T @ Z @ K2
    ZZT = np.trace(Z @ Z.T)

    invL = 1 / (np.trace(K1TK1) * np.trace(K2TK2) + alpha)
    factor = 1 - alpha * invL


    Y = S
    tstep = 1
    lastRes = np.inf

    for iter in range(100000):
        term2 = K1TZK2 - K1TK1 @ Y @ K2TK2
        Snew = factor * Y + invL * term2
        Snew[Snew<0] = 0

        tnew = 0.5 * (1 + np.sqrt(1 + 4 * tstep**2))
        tRatio = (tstep - 1) / tnew
        Y = Snew + tRatio * (Snew - S)
        tstep = tnew
        S = Snew

        if iter % 500 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            ObjFunc = ZZT - 2 * np.trace(S.T @ K1TZK2) + np.trace(S.T @ K1TK1 @ S @ K2TK2) + TikhTerm

            Res = np.abs(ObjFunc - lastRes) / ObjFunc
            lastRes = ObjFunc
            print(f'# It = {iter} >>> Obj. Func. = {ObjFunc:.4f} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S

def plot_Z(tau1, tau2, Z, Out):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25, 10))

    ax1.scatter(tau1, Z[:, 0], s = 30)
    ax1.set_xlabel(r'$\tau_1$ [ms]')
    ax1.set_ylabel('SR')

    ax2.scatter(tau2, Z[-1, :], s = 2)
    ax2.set_xlabel(r'$\tau_2$ [ms]')
    ax2.set_ylabel('CPMG')

    plt.savefig(f'{Out}-DomTemp')

def plot_proj(T1, T2, S, Out):
    projT1 = np.sum(S, axis=1)
    # peaks1, _ = find_peaks(projT1, height=0.1)
    peaks1, _ = find_peaks(projT1, height=0.05)
    peaks1x, peaks1y = T1[peaks1], projT1[peaks1]
    projT2 = np.sum(S, axis=0)
    # peaks2, _ = find_peaks(projT2, height=0.1)
    peaks2, _ = find_peaks(projT2, height=0.1)
    peaks2x, peaks2y = T2[peaks2], projT2[peaks2]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25, 10))

    if np.max(peaks1y) < np.max(projT1)/4:
        ymax = 1.1 * np.max(peaks1y)
    else:
        ymax = None

    ax1.plot(T1, projT1)
    ax1.plot(peaks1x, peaks1y, lw = 0, marker=2, color='black')
    # for i in range(len(peaks1x)):
    #     ax1.annotate(f'({peaks1x[i]:.2f}, {peaks1y[i]:.2f})', xy = (peaks1x[i], peaks1y[i]), fontsize=30)
    ax1.set_xlabel(r'$T_1$ [ms]')
    ax1.set_xscale('log')
    # ax1.set_ylim(bottom=-0.005, top=ymax)
    ax1.set_ylim(bottom=-0.05, top=ymax)
    # ax1.set_ylim(bottom=-0.5, top=ymax)

    if np.max(peaks2y) < np.max(projT2)/4:
        ymax = 1.1 * np.max(peaks2y)
    else:
        ymax = None

    ax2.plot(T2, projT2)
    ax2.plot(peaks2x, peaks2y, lw = 0, marker=2, color='black')
        # for i in range(len(peaks2x)):
        #     ax2.annotate(f'({peaks2x[i]:.2f}, {peaks2y[i]:.2f})', xy = (peaks2x[i], peaks2y[i]), fontsize=30)
    ax2.set_xlabel(r'$T_2$ [ms]')
    ax2.set_xscale('log')
    # ax2.set_ylim(bottom=-0.005, top=ymax)
    ax2.set_ylim(bottom=-0.05, top=ymax)
    # ax2.set_ylim(bottom=-0.5, top=ymax)

    plt.savefig(f'{Out}-DomRates1D')
    np.savetxt(f"{Out}-DomRates1D_T1.csv", projT1)
    np.savetxt(f"{Out}-DomRates1D_T2.csv", projT1)

    return peaks1x, peaks2x

def plot_map(T1, T2, S, nLevel, Out, peaks1x, peaks2x, T1min, T1max, T2min, T2max, RGnorm, alpha, Back, m):
    mini = np.max([T1min, T2min])
    maxi = np.min([T1max, T2max])

    fig, ax = plt.subplots()

    fig.suptitle(f'RG = {RGnorm} dB | Alpha = {alpha} \n BG = {Back} | m = {m}', fontsize='small')

    ax.plot([10.0**mini, 10.0**maxi], [10.0**mini, 10.0**maxi], color='black', ls='-', alpha=0.7, zorder=-2, label = r'$T_1$ = $T_2$')
    for i in range(len(peaks1x)):
        ax.hlines(peaks1x[i], xmin = 10.0**T2min , xmax = 10.0**T2max, color='gray', ls=':')
        ax.annotate(f'   {peaks1x[i]:.2f}', xy = (10.0**T2max, peaks1x[i]), fontsize=15)
    for i in range(len(peaks2x)):
        ax.vlines(peaks2x[i], ymin = 10.0**T1min , ymax = 10.0**T1max, color='gray', ls=':')
        ax.annotate(f'   {peaks2x[i]:.2f}', xy = (peaks2x[i], 10.0**T1max), fontsize=15, rotation = 60)

    ax.contour(T2[15:-15], T1[15:-15], S[15:-15, 15:-15], nLevel, cmap='rainbow')
    ax.set_xlabel(r'$T_2$ [ms]')
    ax.set_ylabel(r'$T_1$ [ms]')
    ax.set_xlim(10.0**T2min, 10.0**T2max)
    ax.set_ylim(10.0**T1min, 10.0**T1max)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right')

    plt.savefig(f'{Out}-DomRates2D')

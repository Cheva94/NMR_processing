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

plt.rcParams["figure.figsize"] = 12.5, 13.5
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linestyle"] = '-'

def CPMG_file(File, Out, T2min, T2max, niniT2):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()
    tau = data[:, 0] # In ms
    nP = len(tau)

    Re = data[:, 1]
    Im = data[:, 2]
    decay = Re + Im * 1j # Complex signal

    nBin = 150
    S0 = np.ones(nBin)
    T2 = np.logspace(T2min, T2max, nBin)
    K = np.zeros((nP, nBin))

    for i in range(nP):
        K[i, :] = np.exp(-tau[i] / T2)

    pAcq = pd.read_csv(File.split(".txt")[0]+'-acqs.txt', header = None, delim_whitespace = True)
    nS, RG, p90, att, RD, tEcho, nEcho = pAcq.iloc[0, 1], pAcq.iloc[1, 1], pAcq.iloc[2, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1], 2*pAcq.iloc[6, 1], pAcq.iloc[7, 1]

    return S0, T2, tau[niniT2:], K, decay[niniT2:], nS, RG, p90, att, RD, tEcho, nEcho

def PhCorr(decay):
    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        decay_ph = decay * np.exp(1j * tita)
        initVal[i] = decay_ph[0].real

    decay = decay * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))

    return decay.real

def Norm(Z, RGnorm, RG, m):
    if RGnorm == "off":
        Norm = 1 / m
    elif RGnorm == 'on':
        Norm = 1 / ((6.32589E-4 * np.exp(RG/9) - 0.0854) * m)
    return Z * Norm

def NLI_FISTA(K, Z, alpha, S):
    Z = np.reshape(Z, (len(Z), 1))
    S = np.reshape(S, (len(S), 1))

    KTK = K.T @ K
    KTZ = K.T @ Z
    ZZT = np.trace(Z @ Z.T)

    invL = 1 / (np.trace(KTK) + alpha)
    factor = 1 - alpha * invL

    Y = S
    tstep = 1
    lastRes = np.inf

    for iter in range(100000):
        term2 = KTZ - KTK @ Y
        Snew = factor * Y + invL * term2
        Snew[Snew<0] = 0

        tnew = 0.5 * (1 + np.sqrt(1 + 4 * tstep**2))
        tRatio = (tstep - 1) / tnew
        Y = Snew + tRatio * (Snew - S)
        tstep = tnew
        S = Snew

        if iter % 500 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            ObjFunc = ZZT - 2 * np.trace(S.T @ KTZ) + np.trace(S.T @ KTK @ S) + TikhTerm

            Res = np.abs(ObjFunc - lastRes) / ObjFunc
            lastRes = ObjFunc
            print(f'# It = {iter} >>> Obj. Func. = {ObjFunc:.4f} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S[:, 0]

def plot_Z(tau, Z, Out, nS, RG, RGnorm, p90, att, RD, tEcho, nEcho, Back, m):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    fig.suptitle(f'nS={nS} | RG = {RG} dB ({RGnorm}) | RD = {RD} s | p90 = {p90} us | Atten = {att} dB | Ecos = {nEcho} | tE = {tEcho} ms | BG = {Back} | m = {m}', fontsize='small')
    ax1.plot(tau, Z)
    ax1.set_xlabel(r'$\tau$ [ms]')
    ax1.set_ylabel('M')

    ax2.semilogy(tau, Z)
    ax2.set_xlabel(r'$\tau$ [ms]')
    ax2.set_ylabel('log(M)')

    plt.savefig(f'{Out}-DomTemp')

def plot_spec(T2, S, Out, alpha, nS, RG, RGnorm, p90, att, RD, tEcho, nEcho, Back, m):
    peaks, _ = find_peaks(S, distance = 20, height = 0.05)
    peaksx, peaksy = T2[peaks], S[peaks]

    fig, ax = plt.subplots()
    
    fig.suptitle(f'nS={nS} | RG = {RG} dB ({RGnorm}) | RD = {RD} s | p90 = {p90} us | Atten = {att} dB | Ecos = {nEcho} | tE = {tEcho} ms | Alpha = {alpha} | BG = {Back} | m = {m}', fontsize='small')

    ax.plot(T2, S)
    ax.plot(peaksx, peaksy, lw = 0, marker=2, color='black')
    for i in range(len(peaksx)):
        ax.annotate(f'({peaksx[i]:.2f}, {peaksy[i]:.2f})', xy = (peaksx[i], peaksy[i]), fontsize=15)
    ax.set_xlabel(r'$T_2$ [ms]')
    ax.set_xscale('log')

    plt.savefig(f'{Out}-DomRates')

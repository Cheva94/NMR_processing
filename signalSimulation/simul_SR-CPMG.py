#!/usr/bin/python3.10

import numpy as np
np.random.seed(1994)
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 30

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1.5

plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1.5

plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.fontsize"] = 30

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4

# Colores
Verde = '#08a189' # = 08A189 = 8 161 137    (principal)
Naranja = '#fa6210' # = FA6210 = 250 98 16    (secundario)
Azul = '#3498db' # (terciario)
Morado = '#6a1b9a' # (cuaternario)
Gris = '#808080' # (alternativo)
Negro = '#000000' # =  = 0 0 0 (base)

def NLI_FISTA_2D(K1, K2, Z, alpha, S):
    '''
    Numeric Laplace inversion, based on FISTA.
    '''

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

        if iter % 1000 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            ObjFunc = ZZT - 2 * np.trace(S.T @ K1TZK2) + np.trace(S.T @ K1TK1 @ S @ K2TK2) + TikhTerm

            Res = np.abs(ObjFunc - lastRes) / ObjFunc
            lastRes = ObjFunc
            print(f'\t# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S, iter

################################################################################

alpha = 0.1
T1s = np.array([150, 900]) # ms
T2s = np.array([150, 900]) # ms
Amps = np.array([0.3, 0.7])
T1min, T1max = 2, 4
T2min, T2max = 2, 4

largo1 = 20
largo2 = 100
tau1 = np.logspace(0, 4, largo1) # ms
nP1 = len(tau1)
tau2 = np.logspace(0, 4, largo2) # ms
nP2 = len(tau2)

SGL0 = np.zeros((nP1, nP2))
SGL1 = np.zeros((nP1, nP2))
for i in range(nP1):
    for j in range(nP2):
        SGL0[i, j] = Amps[0] * (1 - np.exp(-tau1[i]/T1s[0])) * np.exp(-tau2[j]/T2s[0])
        SGL1[i, j] = Amps[1] * (1 - np.exp(-tau1[i]/T1s[1])) * np.exp(-tau2[j]/T2s[1])
SGLnoisynt = SGL0 + SGL1

noise = np.random.normal(0, 0.007, (largo1, largo2))
SGLnoisy = SGLnoisynt + noise

nBinx = nBiny = 300
S0 = np.ones((nBinx, nBiny))
T1 = np.logspace(T1min, T1max, nBinx)
T2 = np.logspace(T2min, T2max, nBiny)

K1 = np.zeros((nP1, nBinx))
for i in range(nP1):
    K1[i, :] = 1 - np.exp(-tau1[i] / T1)
K2 = np.zeros((nP2, nBiny))
for i in range(nP2):
    K2[i, :] = np.exp(-tau2[i] / T2)


Snoisy, _ = NLI_FISTA_2D(K1, K2, SGLnoisy, alpha, S0)
np.savetxt('simul_SR-CPMG_conRuido_Laplace.txt', Snoisy)
# Snoisy = pd.read_csv('simul_SR-CPMG_conRuido_Laplace.txt', header = None, delim_whitespace=True).to_numpy()

_, axs = plt.subplots(subplot_kw={"projection": "3d"})
tau2, tau1 = np.meshgrid(tau2, tau1)
axs.plot_surface(tau1, tau2, SGLnoisy, cmap='viridis', rcount=100, ccount=100)

axs.set_xlabel(r'$\tau_1$ [ms]')
axs.set_ylabel(r'$\tau_2$ [ms]')
axs.set_zlabel('Señal SR-CPMG')

axs.xaxis.labelpad=30
axs.yaxis.labelpad=30
axs.zaxis.labelpad=30

axs.set_zlim(0, 1)
axs.zaxis.set_major_locator(MultipleLocator(0.2))
axs.zaxis.set_minor_locator(MultipleLocator(0.1))

axs.xaxis.set_major_locator(MultipleLocator(3000))
axs.xaxis.set_minor_locator(MultipleLocator(1500))

axs.yaxis.set_major_locator(MultipleLocator(3000))
axs.yaxis.set_minor_locator(MultipleLocator(1500))

axs.view_init(elev=20, azim=135)

plt.savefig('simul_SR-CPMG_conRuido_TimeDomain')

_, axs = plt.subplots()
Snorm = Snoisy / np.max(Snoisy)

# Remove edge effects from NLI on the plots
Snorm = Snorm[4:-9, 2:-2]
T1 = T1[4:-9]
T2 = T2[2:-2]
Snorm = np.where(Snorm<0.05, 0.0, Snorm)

axs.contour(T2, T1, Snorm, 100, cmap='rainbow')
axs.set_xlabel(r'$T_2$ [ms]')
axs.set_ylabel(r'$T_1$ [ms]')
axs.set_xlim(10.0**T2min, 10.0**T2max)
axs.set_ylim(10.0**T1min, 10.0**T1max)
axs.set_xscale('log')
axs.set_yscale('log')

plt.savefig('simul_SR-CPMG_conRuido_Laplace')
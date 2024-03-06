#!/uCPMG/bin/python3.10

import numpy as np
np.random.seed(1994)
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

def NLI_FISTA_1D(K, Z, alpha, S):
    '''
    Numeric Laplace inversion, based on FISTA.
    '''

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
            print(f'\t# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S[:, 0], iter

def fitLapMag_1D(tau2, T2, S):
    '''
    Fits decay from T2 distribution.
    '''

    M = []
    for i in range(len(tau2)):
        m = 0
        for j in range(len(T2)):
            m += S[j] * np.exp(-tau2[i] / T2[j])
        M.append(m)

    return M

################################################################################

largo = 20
tau2 = np.logspace(0, 3, largo) # ms
nP = len(tau2)
T2s = np.array([10, 100]) # ms
T2min, T2max = 0.5, 2.5
Amps = np.array([0.3, 0.7])
SGL = Amps[0] * np.exp(-tau2/T2s[0]) + Amps[1] * np.exp(-tau2/T2s[1])
noise = np.random.normal(0, 0.015, largo)
SGLnoisy = SGL + noise

nBin = 300
S0 = np.ones(nBin)
T2 = np.logspace(T2min, T2max, nBin)
K = np.zeros((nP, nBin))
for i in range(nP):
    K[i, :] = np.exp(-tau2[i] / T2)

alpha = 0.003
Snoisy, iter = NLI_FISTA_1D(K, SGLnoisy, alpha, S0)

MagLap = fitLapMag_1D(tau2, T2, Snoisy)

# Remove edge effects from NLI on the plots
Snoisy = Snoisy[4:]
T2 = T2[4:]

_, axs = plt.subplots()

axs.plot(tau2, SGLnoisy, color=Negro, marker='*', ms=20, ls='', label='Dato')
axs.plot(tau2, MagLap, color=Azul, ls=':', zorder=-10, label='Ajuste')
axs.set_xlabel(r'$\tau_2$ [ms]')
axs.set_ylabel('Señal CPMG')
axs.legend(loc='upper right')

axs.xaxis.set_major_locator(MultipleLocator(200))
axs.xaxis.set_minor_locator(MultipleLocator(100))
axs.yaxis.set_major_locator(MultipleLocator(0.2))
axs.yaxis.set_minor_locator(MultipleLocator(0.1))

plt.savefig('simul_CPMG_TimeDomain')

Snorm = Snoisy / np.max(Snoisy)
peaks, _ = find_peaks(Snorm,height=0.025, distance = 5)
peaksx, peaksy = T2[peaks], Snorm[peaks]

_, axs = plt.subplots()

axs.plot(T2, Snorm, color = Verde)
for i in range(len(peaksx)):
    axs.plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, ms=20, color=Naranja)
    axs.annotate(f'{peaksx[i]:.0f} ms', xy = (peaksx[i], peaksy[i] + 0.07), 
                    fontsize=30, ha='center')
axs.set_xlabel(r'$T_2$ [ms]')
axs.set_xscale('log')
axs.set_ylim(-0.02, 1.2)
axs.set_xlim(10.0**T2min, 10.0**T2max)

axs.yaxis.set_major_locator(MultipleLocator(0.2))
axs.yaxis.set_minor_locator(MultipleLocator(0.1))

plt.savefig('simul_CPMG_Laplace')
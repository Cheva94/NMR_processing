import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from scipy.signal import find_peaks
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit

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

plt.rcParams["figure.figsize"] = 50, 20
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'


def plot(tau1, tau2, Z, T1, T2, S, M1, M2, Out, T1min, T1max, T2min, T2max, alpha, cropT1, cropT2, Map, nS, RDT, RG, att, RD, p90, p180, tE, nE):
    '''
    Grafica resultados.
    '''

    fig, axs = plt.subplots(2,4)
    if Map != 'fid':
        fig.suptitle(rf'nS={nS}    |    RDT = {RDT} ms    |    RG = {RG} dB    |    Atten = {att} dB    |    RD = {RD} s    |    p90 = {p90} $\mu$s    |    p180 = {p180} $\mu$s    |    tE = {tE} ms    |    Ecos = {nE}', fontsize='large')
    else:
        fig.suptitle(rf'nS={nS}    |    RDT = {RDT} ms    |    RG = {RG} dB    |    Atten = {att} dB    |    RD = {RD} s    |    p90 = {p90} $\mu$s', fontsize='large')

    NegPts = 0
    for k in range(len(tau1)):
        if Z[k, 0] < 0.0000:
            NegPts += 1

    # SR: experimental y ajustada (Laplace 2D)
    projOffset = Z[-1, 0] / M1[-1]
    M1 *= projOffset

    axs[0,0].set_title(f'Pts. desc.: {cropT1:.0f}; Neg. pts.: {NegPts}', fontsize='large')
    axs[0,0].scatter(tau1, Z[:, 0], label='Exp', color='coral', zorder=5)
    axs[0,0].plot(tau1, M1, label='2D-Lap', color='teal', zorder=0)
    axs[0,0].set_xlabel(r'$\tau_1$ [ms]')
    axs[0,0].set_ylabel('SR')
    axs[0,0].legend()
    
    # SR: residuos
    residuals = M1-Z[:, 0]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z[:, 0] - np.mean(Z[:, 0])) ** 2)
    R2_indir = 1 - ss_res / ss_tot

    axs[1,0].set_title(fr'Res. dim. indir.: R$^2$ = {R2_indir:.6f}', fontsize='large')
    axs[1,0].scatter(tau1, M1-Z[:, 0], color = 'blue')
    axs[1,0].axhline(0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(-0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,0].set_xlabel(r'$\tau$1 [ms]')

    # CPMG/FID/FID-CPMG: experimental y ajustada (Laplace 2D)
    axs[0,1].set_title(f'Pts. desc.: {cropT2:.0f}', fontsize='large')
    axs[0,1].scatter(tau2, Z[-1, :], label='Exp', color='coral')
    axs[0,1].plot(tau2, M2, label='2D-Lap', color='teal')
    axs[0,1].legend()
    axs[0,1].axhline(0, c = 'k', lw = 4, ls = ':', zorder=-2)

    # Inset del comienzo de la CPMG/FID/FID-CPMG:
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=5)
    axins2.scatter(tau2[0:20], Z[-1, :][0:20], color='coral')
    axins2.plot(tau2[0:20], M2[0:20], color='teal')

    # CPMG/FID/FID-CPMG: residuos
    residuals = M2-Z[-1, :]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z[-1, :] - np.mean(Z[-1, :])) ** 2)
    R2_dir = 1 - ss_res / ss_tot

    axs[1,1].set_title(fr'Res. dim. dir.: R$^2$ = {R2_dir:.6f}', fontsize='large')
    axs[1,1].scatter(tau2, M2-Z[-1, :], color = 'blue')
    axs[1,1].axhline(0, c = 'k', lw = 4, ls = '-')
    axs[1,1].axhline(0.1*np.max(Z[-1,:]), c = 'red', lw = 6, ls = '-')
    axs[1,1].axhline(-0.1*np.max(Z[-1,:]), c = 'red', lw = 6, ls = '-')

    # Distribución proyectada de T1
    T1 = T1[4:-9]
    projT1 = np.sum(S, axis=1)
    projT1 = projT1[4:-9] / np.max(projT1[4:-9])
    peaks1, _ = find_peaks(projT1, height=0.025, distance = 5)
    peaks1x, peaks1y = T1[peaks1], projT1[peaks1]

    cumT1 = np.cumsum(projT1)
    cumT1 /= cumT1[-1]

    axs[0,2].set_title(f'Proyección de T1 del mapa', fontsize='large')
    axs[0,2].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,2].plot(T1, projT1, label = 'Distrib.', color = 'teal')
    for i in range(len(peaks1x)):
        axs[0,2].plot(peaks1x[i], peaks1y[i] + 0.05, lw = 0, marker=11, color='black')
        axs[0,2].annotate(f'{peaks1x[i]:.2f}', xy = (peaks1x[i], peaks1y[i] + 0.07), fontsize=30, ha = 'center')
    axs[0,2].set_xlabel(r'$T_1$ [ms]')
    axs[0,2].set_ylabel(r'Distrib. $T_1$')
    axs[0,2].set_xscale('log')
    axs[0,2].set_xlim(10.0**T1min, 10.0**T1max)
    axs[0,2].set_ylim(-0.02, 1.2)

    ax = axs[0,2].twinx()
    ax.plot(T1, cumT1, label = 'Cumul.', color = 'coral')
    ax.set_ylim(-0.02, 1.2)
    ax.set_ylabel(r'Cumul. $T_1$')

    # Distribución proyectada de T2
    T2 = T2[2:-2]
    projT2 = np.sum(S, axis=0)
    projT2 = projT2[2:-2] / np.max(projT2[2:-2])
    peaks2, _ = find_peaks(projT2, height=0.025, distance = 5)
    peaks2x, peaks2y = T2[peaks2], projT2[peaks2]

    cumT2 = np.cumsum(projT2)
    cumT2 /= cumT2[-1]

    axs[0,3].set_title(f'Proyección de T2 del mapa', fontsize='large')
    axs[0,3].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,3].plot(T2, projT2, label = 'Distrib.', color = 'teal')
    for i in range(len(peaks2x)):
        axs[0,3].plot(peaks2x[i], peaks2y[i] + 0.05, lw = 0, marker=11, color='black')
        axs[0,3].annotate(f'{peaks2x[i]:.2f}', xy = (peaks2x[i], peaks2y[i] + 0.07), fontsize=30, ha = 'center')
    axs[0,3].set_xscale('log')
    axs[0,3].set_ylim(-0.02, 1.2)
    axs[0,3].set_xlim(10.0**T2min, 10.0**T2max)

    ax = axs[0,3].twinx()
    ax.plot(T2, cumT2, label = 'Cumul.', color = 'coral')
    ax.set_ylim(-0.02, 1.2)

    # Mapa T1-T2
    mini = np.max([T1min, T2min])
    maxi = np.min([T1max, T2max])
    S = S[4:-9, 2:-2]

    axs[1,3].set_title(rf'$\alpha$ = {alpha}')
    axs[1,3].plot([10.0**mini, 10.0**maxi], [10.0**mini, 10.0**maxi], color='black', ls='-', alpha=0.7, zorder=-2, label = r'$T_1$ = $T_2$')
    for i in range(len(peaks2x)):
        axs[1,3].axvline(x=peaks2x[i], color='k', ls=':', lw=4)
    for i in range(len(peaks1x)):
        axs[1,3].axhline(y=peaks1x[i], color='k', ls=':', lw=4)
    axs[1,3].contour(T2, T1, S, 100, cmap='rainbow')
    axs[1,3].set_ylabel(r'$T_1$ [ms]')
    axs[1,3].set_xlim(10.0**T2min, 10.0**T2max)
    axs[1,3].set_ylim(10.0**T1min, 10.0**T1max)
    axs[1,3].set_xscale('log')
    axs[1,3].set_yscale('log')
    axs[1,3].legend(loc='lower right')

    axs[0,1].set_xlabel(r'$\tau_2$ [ms]')
    axs[0,1].set_ylabel('CPMG')

    axs[1,1].set_xlabel(r'$\tau_2$ [ms]')

    axs[0,3].set_xlabel(r'$T_2$ [ms]')
    axs[0,3].set_ylabel(r'Distrib. $T_2$')
    axs[0,3].fill_between([tE, 5 * tE], -0.02, 1.2, color='red', alpha=0.3, zorder=-2)

    ax.set_ylabel(r'Cumul. $T_2$')

    axs[1,3].set_xlabel(r'$T_2$ [ms]')
    axs[1,3].fill_between([tE, 5 * tE], 10.0**T1min, 10.0**T1max, color='red', alpha=0.3, zorder=-2)

    plt.savefig(f'{Out}')
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

def DT2_file(File, Dmin, Dmax, T2min, T2max, cropD, cropT2):
    '''
    Lectura del archivo de la medición.
    '''

    signal = pd.read_csv(File, header = None, sep = '\t').to_numpy()

    Nx = Ny = 150
    S0 = np.ones((Nx, Ny))
    S01D = np.ones(Nx)
    D = np.logspace(Dmin, Dmax, Nx)
    T2 = np.logspace(T2min, T2max, Ny)

    tau1 = pd.read_csv('DiffAxis.dat', header = None, delim_whitespace = True).to_numpy()
    tau2 = pd.read_csv('T2Axis.dat', header = None, delim_whitespace = True).to_numpy()

    N1, N2 = len(tau1), len(tau2)

    K1 = np.exp(-tau1 * D)
    K2 = np.exp(-tau2 / T2)

    return S0, D, T2, tau1, tau2, K1, K2, signal, N1, N2


def NLI_FISTA_2D(K1, K2, Z, alpha, S):
    '''
    Inversión de Laplace 2D
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
            print(f'# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break
    return S

def fitMag_2D(tau1, tau2, D, T2, S):
    '''
    Ajuste de los decaimientos a partir de la distribución de D y T2.
    '''

    print(f'Fitting D projection from 2D-Laplace in time domain...')

    t1 = range(len(tau1))
    d1 = range(len(D))
    S1 = np.sum(S, axis=1)
    M1 = []

    for i in t1:
        m1 = 0
        for j in d1:
            m1 += S1[j] * (np.exp(-tau1[i] * D[j]))
        M1.append(m1[0])

    print(f'Fitting T2 projection from 2D-Laplace in time domain...')

    t2 = range(len(tau2))
    d2 = range(len(T2))
    S2 = np.sum(S, axis=0)
    M2 = []

    for i in t2:
        m2 = 0
        for j in d2:
            m2 += S2[j] * np.exp(-tau2[i] / T2[j])
        M2.append(m2[0])

    return np.array(M1), np.array(M2)

def plot(tau1, tau2, Z, D, T2, S_2D, M1, M2, Out, Dmin, Dmax, T2min, T2max, alpha):
    '''
    Grafica resultados.
    '''

    fig, axs = plt.subplots(2,4)

    # SR: experimental y ajustada (Laplace 2D)
#    projOffset = Z[-1, 0] / M1[-1]
    projOffset = Z[-1, -1] / M1[-1]
    M1 *= projOffset

#    axs[0,0].scatter(tau1, Z[:, 0], label='Exp', color='coral', zorder=5)
    axs[0,0].scatter(tau1, Z[:, -1], label='Exp', color='coral', zorder=5)
    axs[0,0].plot(tau1, M1, label='2D-Lap', color='teal', zorder=0)
    axs[0,0].set_xlabel(r'$\tau_1$ [ms]')
    axs[0,0].set_ylabel('D [10-9 m2/s]')
    axs[0,0].legend()

    # SR: residuos
#    residuals = M1-Z[:, 0]
#    ss_res = np.sum(residuals ** 2)
#    ss_tot = np.sum((Z[:, 0] - np.mean(Z[:, 0])) ** 2)
#    R2_indir = 1 - ss_res / ss_tot

#    axs[1,0].set_title(fr'Res. dim. indir.: R$^2$ = {R2_indir:.6f}', fontsize='large')
#    axs[1,0].scatter(tau1, M1-Z[:, 0], color = 'blue')
    axs[1,0].scatter(tau1, M1-Z[:, -1], color = 'blue')
#    axs[1,0].axhline(0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
#    axs[1,0].axhline(-0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(0.1*np.max(Z[:, -1]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(-0.1*np.max(Z[:, -1]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,0].set_xlabel(r'$\tau$1 [ms]')

    # CPMG/FID/FID-CPMG: experimental y ajustada (Laplace 2D)
#    axs[0,1].scatter(tau2, Z[-1, :], label='Exp', color='coral')
    axs[0,1].scatter(tau2, Z[0, :], label='Exp', color='coral')
    axs[0,1].plot(tau2, M2, label='2D-Lap', color='teal')
    axs[0,1].legend()
    axs[0,1].axhline(0, c = 'k', lw = 4, ls = ':', zorder=-2)

    # Inset del comienzo de la CPMG/FID/FID-CPMG:
    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=5)
#    axins2.scatter(tau2[0:20], Z[-1, :][0:20], color='coral')
    axins2.scatter(tau2[0:20], Z[0, :][0:20], color='coral')
    axins2.plot(tau2[0:20], M2[0:20], color='teal')

    # CPMG/FID/FID-CPMG: residuos
#    residuals = M2-Z[-1, :]
#    ss_res = np.sum(residuals ** 2)
#    ss_tot = np.sum((Z[-1, :] - np.mean(Z[-1, :])) ** 2)
#    R2_dir = 1 - ss_res / ss_tot

#    axs[1,1].set_title(fr'Res. dim. dir.: R$^2$ = {R2_dir:.6f}', fontsize='large')
#    axs[1,1].scatter(tau2, M2-Z[-1, :], color = 'blue')
    axs[1,1].scatter(tau2, M2-Z[0, :], color = 'blue')
    axs[1,1].axhline(0, c = 'k', lw = 4, ls = '-')
#    axs[1,1].axhline(0.1*np.max(Z[-1,:]), c = 'red', lw = 6, ls = '-')
#    axs[1,1].axhline(-0.1*np.max(Z[-1,:]), c = 'red', lw = 6, ls = '-')
    axs[1,1].axhline(0.1*np.max(Z[0,:]), c = 'red', lw = 6, ls = '-')
    axs[1,1].axhline(-0.1*np.max(Z[0,:]), c = 'red', lw = 6, ls = '-')

    # Distribución proyectada de D
    D = D[4:-9]
    projD = np.sum(S_2D, axis=1)
    projD = projD[4:-9] / np.max(projD[4:-9])
    peaks1, _ = find_peaks(projD, height=0.025, distance = 5)
    peaks1x, peaks1y = D[peaks1], projD[peaks1]

    cumD = np.cumsum(projD)
    cumD /= cumD[-1]

    axs[0,2].set_title(f'Proyección de D del mapa', fontsize='large')
    axs[0,2].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,2].plot(D, projD, label = 'Distrib.', color = 'teal')
    for i in range(len(peaks1x)):
        axs[0,2].plot(peaks1x[i], peaks1y[i] + 0.05, lw = 0, marker=11, color='black')
        axs[0,2].annotate(f'{peaks1x[i]:.2f}', xy = (peaks1x[i], peaks1y[i] + 0.07), fontsize=30, ha = 'center')
    axs[0,2].set_xlabel(r'D [10-9 m2/s]')
    axs[0,2].set_ylabel(r'Distrib. $T_1$')
    axs[0,2].set_xscale('log')
    axs[0,2].set_xlim(10.0**Dmin, 10.0**Dmax)
    axs[0,2].set_ylim(-0.02, 1.2)

    ax = axs[0,2].twinx()
    ax.plot(D, cumD, label = 'Cumul.', color = 'coral')
    ax.set_ylim(-0.02, 1.2)
    ax.set_ylabel(r'Cumul. $T_1$')

    # Distribución proyectada de T2
    T2 = T2[2:-2]
    projT2 = np.sum(S_2D, axis=0)
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

    # Mapa D-T2
    mini = np.max([Dmin, T2min])
    maxi = np.min([Dmax, T2max])
    S_2D = S_2D[4:-9, 2:-2]

    axs[1,3].set_title(rf'$\alpha$ = {alpha}')
 #   axs[1,3].plot([10.0**mini, 10.0**maxi], [10.0**mini, 10.0**maxi], color='black', ls='-', alpha=0.7, zorder=-2, label = r'$T_1$ = $T_2$')
    for i in range(len(peaks2x)):
        axs[1,3].axvline(x=peaks2x[i], color='k', ls=':', lw=4)
    for i in range(len(peaks1x)):
        axs[1,3].axhline(y=peaks1x[i], color='k', ls=':', lw=4)
    axs[1,3].contour(T2, D, S_2D, 100, cmap='rainbow')
    axs[1,3].set_ylabel(r'D [10-9 m2/s]')
    axs[1,3].set_xlim(10.0**T2min, 10.0**T2max)
    axs[1,3].set_ylim(10.0**Dmin, 10.0**Dmax)
    axs[1,3].set_xscale('log')
    axs[1,3].set_yscale('log')
#    axs[1,3].legend(loc='lower right')

    axs[0,1].set_xlabel(r'$\tau_2$ [ms]')
    axs[0,1].set_ylabel('FID')

    axs[1,1].set_xlabel(r'$\tau_2$ [ms]')

    axs[0,3].set_xlabel(r'$T_2$ [ms]')
    axs[0,3].set_ylabel(r'Distrib. $T_2$')

    ax.set_ylabel(r'Cumul. $T_2$')

    axs[1,3].set_xlabel(r'$T_2$ [ms]')

    plt.savefig(f'{Out}')

    print('Writing output...')
    np.savetxt(f"{Out}-DomRates2D.csv", S_2D, delimiter='\t')

    with open(f'{Out}-DomRates2D_D.csv', 'w') as f:
        f.write("D [ms], Distribution, Cumulative \n")
        for i in range(len(D)):
            f.write(f'{D[i]:.6f}, {projD[i]:.6f}, {cumD[i]:.6f} \n')

    with open(f'{Out}-DomRates2D_T2.csv', 'w') as f:
        f.write("T2 [ms], Distribution, Cumulative \n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}, {projT2[i]:.6f}, {cumT2[i]:.6f} \n')

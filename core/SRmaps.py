import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from scipy.signal import find_peaks
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from scipy.ndimage import gaussian_filter1d
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

def SRmap_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2, Map):
    '''
    Lectura del archivo de la medición.
    '''

    data = pd.read_csv(File, header = None, delim_whitespace = True).to_numpy()
    Re = data[:, 0]
    Im = data[:, 1]
    signal = Re + Im * 1j # Complex signal

    pAcq = pd.read_csv(File.split(".txt")[0]+'_acqs.txt', header = None, sep='\t')
    nS, RDT, RG, att, RD, p90 = pAcq.iloc[0, 1], pAcq.iloc[1, 1], pAcq.iloc[2, 1], pAcq.iloc[3, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1]

    if Map == 'fid':
        p180 = tE = nE = None
        nFID = 0
    elif Map == 'fidcpmg':
        p180, tE, nE = pAcq.iloc[6, 1], pAcq.iloc[10, 1], pAcq.iloc[11, 1]
        nFID = 0
    elif Map == 'cpmg':
        p180, tE, nE = pAcq.iloc[6, 1], pAcq.iloc[10, 1], pAcq.iloc[11, 1]
        nFID = 1250 * tE - 54

    niniT2new = niniT2 + nFID

    Nx = Ny = 150
    S0 = np.ones((Nx, Ny))
    S01D = np.ones(Nx)
    T1 = np.logspace(T1min, T1max, Nx)
    T2 = np.logspace(T2min, T2max, Ny)

    tau1 = pd.read_csv(File.split('.txt')[0]+"_t1.dat", header = None, delim_whitespace = True).to_numpy()
    tau2 = pd.read_csv(File.split('.txt')[0]+"_t2.dat", header = None, delim_whitespace = True).to_numpy()
    N1, N2 = len(tau1), len(tau2)
    tau1 = tau1[niniT1:]
    tau2 = tau2[niniT2new:]

    K1 = 1 - np.exp(-tau1 / T1)
    K2 = np.exp(-tau2 / T2)

    K1D = np.zeros((N1, Nx))
    for i in range(N1):
        K1D[i, :] = 1 - np.exp(-tau1[i] / T1)

    return S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, niniT2new, S01D, K1D

def PhCorr(signal, N1, N2):
    '''
    Corrección de fase, basándose en la última medición.
    '''

    Z = []

    signal_Last = signal[(N1-1)*N2:]
    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal_Last * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    tita = np.deg2rad(max(initVal, key=initVal.get))

    for k in range(N1):
        signal_k = signal[k*N2:(k+1)*N2] * np.exp(1j * tita)
        signal_k = signal_k.real
        Z.append(signal_k)

    return np.array(Z)

def Norm(Z, RG, N1, N2, niniT1, niniT2new):
    '''
    Normalización por ganancia, corrección de offset y creación de matriz.
    '''

    norm = 1 / (6.32589E-4 * np.exp(RG/9) - 0.0854)
    Z = np.reshape(Z*norm, (N1, N2))[niniT1:, niniT2new:]
    offset = np.min(Z[:, 0])
    Z -= offset

    return Z

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

def fitMag_2D(tau1, tau2, T1, T2, S):
    '''
    Ajuste de los decaimientos a partir de la distribución de T1 y T2.
    '''

    print(f'Fitting T1 projection from 2D-Laplace in time domain...')

    t1 = range(len(tau1))
    d1 = range(len(T1))
    S1 = np.sum(S, axis=1)
    M1 = []

    for i in t1:
        m1 = 0
        for j in d1:
            m1 += S1[j] * (1 - np.exp(-tau1[i] / T1[j]))
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

    return M1, M2

def NLI_FISTA_1D(K, Z, alpha, S):
    '''
    Inversión de Laplace 1D
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

        if iter % 1000 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            ObjFunc = ZZT - 2 * np.trace(S.T @ KTZ) + np.trace(S.T @ KTK @ S) + TikhTerm

            Res = np.abs(ObjFunc - lastRes) / ObjFunc
            lastRes = ObjFunc
            print(f'# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S[:, 0]

def fitMag_1D(tau1, T1, S_1D):
    '''
    Ajuste del decaimiento a partir de la distribución de T1 de la Laplace 1D.
    '''

    t = range(len(tau1))
    d = range(len(T1))
    M = []
    for i in t:
        m = 0
        for j in d:
            m += S_1D[j] * (1 - np.exp(- tau1[i] / T1[j]))
        M.append(m[0])

    return M

def SR1D_exp(tau1, T1, M0):
    out = M0 * (1 - np.exp(-tau1 / T1))
    return out.flatten()

def SR1D_fit(tau1, Z, T1min, T1max):
    '''
    Ajusta la SR aislada.
    '''

    guess_T1 = 10**((T1max-T1min) / 2)
    guess_M0 = (Z[-1] - Z[0]) / 2

    popt, pcov = curve_fit(SR1D_exp, tau1, Z, bounds=(0, np.inf), p0=[guess_T1, guess_M0])
    perr = np.sqrt(np.diag(pcov))

    residuals = Z - SR1D_exp(tau1, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return popt[0], popt[1], perr[0], r2

def plot(tau1, tau2, Z, T1, T2, S, M1, M2, Out, T1min, T1max, T2min, T2max, alpha, Back, niniT1, niniT2, Map, nS, RDT, RG, att, RD, p90, p180, tE, nE, SR1D_T1, SR1D_T1sd, SR1D_r2, SR1D_M0, S_1D, M_1D):
    '''
    Grafica resultados.
    '''

    fig, axs = plt.subplots(2,4)
    if Map != 'fid':
        fig.suptitle(rf'nS={nS:.0f}    |    RDT = {RDT} ms    |    RG = {RG:.0f} dB    |    Atten = {att:.0f} dB    |    RD = {RD:.3f} s    |    p90 = {p90} $\mu$s    |    p180 = {p180} $\mu$s    |    tE = {tE:.1f} ms    |    Ecos = {nE:.0f}', fontsize='large')
    else:
        fig.suptitle(rf'nS={nS:.0f}    |    RDT = {RDT} ms    |    RG = {RG:.0f} dB    |    Atten = {att:.0f} dB    |    RD = {RD:.3f} s    |    p90 = {p90} $\mu$s', fontsize='large')

    # SR: experimental y ajustada (Laplace 2D)
    axs[0,0].set_title(f'{niniT1:.0f} puntos descartados', fontsize='large')
    axs[0,0].scatter(tau1, Z[:, 0], label='Exp', color='coral', zorder=5)
    axs[0,0].plot(tau1, M1, label='Fit 2D-Laplace', color='teal', zorder=0)
    axs[0,0].plot(tau1, SR1D_exp(tau1, SR1D_T1, SR1D_M0), label='Fit Exponencial', color='darkgreen', zorder=1)
    axs[0,0].plot(tau1, M_1D, label='Fit 1D-Laplace', color='navy', zorder=2)
    axs[0,0].set_xlabel(r'$\tau_1$ [ms]')
    axs[0,0].set_ylabel('SR')
    axs[0,0].legend()
    axs[0,0].annotate(fr'R$^2$={SR1D_r2:.6f}', xy = (0.3 * tau1[-1], 0.70 * Z[-1, 0]), fontsize=30, ha = 'left', color='darkgreen')
    axs[0,0].annotate(fr'T$_1$=({SR1D_T1:.2f}$\pm${SR1D_T1sd:.2f})', xy = (0.3 * tau1[-1], 0.60 * Z[-1, 0]), fontsize=30, ha = 'left', color='darkgreen')

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
    axs[0,1].set_title(f'{niniT2:.0f} puntos descartados', fontsize='large')
    axs[0,1].scatter(tau2, Z[-1, :], label='Exp', color='coral')
    axs[0,1].plot(tau2, M2, label='Fit 2D-Laplace', color='teal')
    axs[0,1].legend()

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
    peaks1, _ = find_peaks(projT1, height=0.1, distance = 5)
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
    peaks2, _ = find_peaks(projT2, height=0.1, distance = 5)
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

    axs[1,2].set_title(rf'$\alpha$ = {alpha}')
    axs[1,2].plot([10.0**mini, 10.0**maxi], [10.0**mini, 10.0**maxi], color='black', ls='-', alpha=0.7, zorder=-2, label = r'$T_1$ = $T_2$')
    for i in range(len(peaks2x)):
        axs[1,2].axvline(x=peaks2x[i], color='k', ls=':', lw=4)
    for i in range(len(peaks1x)):
        axs[1,2].axhline(y=peaks1x[i], color='k', ls=':', lw=4)
    axs[1,2].contour(T2, T1, S, 100, cmap='rainbow')
    axs[1,2].set_ylabel(r'$T_1$ [ms]')
    axs[1,2].set_xlim(10.0**T2min, 10.0**T2max)
    axs[1,2].set_ylim(10.0**T1min, 10.0**T1max)
    axs[1,2].set_xscale('log')
    axs[1,2].set_yscale('log')
    axs[1,2].legend(loc='lower right')

    # Distribución de T1 con Laplace 1D
    S_1D = S_1D[4:-9]
    Snorm = S_1D / np.max(S_1D)
    peaks, _ = find_peaks(S_1D)
    peaksx, peaksy = T1[peaks], Snorm[peaks]

    cumT1_1D = np.cumsum(Snorm)
    cumT1_1D /= cumT1_1D[-1]

    residuals = M_1D-Z[:, 0]
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Z[:, 0] - np.mean(Z[:, 0]))**2)
    R2_1D = 1 - ss_res / ss_tot

    axs[1,3].set_title(fr'Laplace 1D: R$^2$ = {R2_1D:.6f}', fontsize='large')
    axs[1,3].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[1,3].plot(T1, Snorm, label = 'Distrib.', color = 'teal')
    for i in range(len(peaksx)):
        if peaksy[i] > 0.1:
            axs[1,3].plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, color='black')
            axs[1,3].annotate(f'{peaksx[i]:.2f}', xy = (peaksx[i], peaksy[i] + 0.07), fontsize=30, ha='center')
    axs[1,3].set_xlabel(r'$T_1$ [ms]')
    axs[1,3].set_ylabel(r'Distrib. $T_1$')
    axs[1,3].set_xscale('log')
    axs[1,3].set_ylim(-0.02, 1.2)
    axs[1,3].set_xlim(10.0**T1min, 10.0**T1max)

    ax1d = axs[1,3].twinx()
    ax1d.plot(T1, cumT1_1D, label = 'Cumul.', color = 'coral')
    ax1d.set_ylim(-0.02, 1.2)
    ax1d.set_ylabel(r'Cumul. $T_1$')

    if Map == 'fid':
        axs[0,1].set_xlabel(r'$\tau_2^*$ [ms]')
        axs[0,1].set_ylabel('FID')

        axs[1,1].set_xlabel(r'$\tau_2^*$ [ms]')

        axs[0,3].set_xlabel(r'$T_2^*$ [ms]')
        axs[0,3].set_ylabel(r'Distrib. $T_2^*$')

        ax.set_ylabel(r'Cumul. $T_2^*$')

        axs[1,2].set_xlabel(r'$T_2^*$ [ms]')

    elif Map == 'cpmg':
        axs[0,1].set_xlabel(r'$\tau_2$ [ms]')
        axs[0,1].set_ylabel('CPMG')

        axs[1,1].set_xlabel(r'$\tau_2$ [ms]')

        axs[0,3].set_xlabel(r'$T_2$ [ms]')
        axs[0,3].set_ylabel(r'Distrib. $T_2$')

        ax.set_ylabel(r'Cumul. $T_2$')

        axs[1,2].set_xlabel(r'$T_2$ [ms]')

    elif Map == 'fidcpmg':
        axs[0,1].set_xlabel(r'$\tau_2^* | \tau_2$ [ms]')
        axs[0,1].set_ylabel('FID-CPMG')

        axs[1,1].set_xlabel(r'$\tau_2^* | \tau_2$ [ms]')

        axs[0,3].set_xlabel(r'$T_2^* | T_2$ [ms]')
        axs[0,3].set_ylabel(r'Distrib. $T_2^* | T_2$')

        ax.set_ylabel(r'Cumul. $T_2^* | T_2$')

        axs[1,2].set_xlabel(r'$T_2^* | T_2$ [ms]')

    plt.savefig(f'{Out}')

    print('Writing output...')
    np.savetxt(f"{Out}-DomRates2D.csv", S, delimiter='\t')
    np.savetxt(f"{Out}-DomRates1D.csv", S_1D, delimiter='\t')

    with open(f'{Out}-DomRates2D_T1.csv', 'w') as f:
        f.write("T1 [ms], Distribution, Cumulative \n")
        for i in range(len(T1)):
            f.write(f'{T1[i]:.6f}, {projT1[i]:.6f}, {cumT1[i]:.6f} \n')

    with open(f'{Out}-DomRates2D_T2.csv', 'w') as f:
        f.write("T2 [ms], Distribution, Cumulative \n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}, {projT2[i]:.6f}, {cumT2[i]:.6f} \n')

    with open(f'{Out}-DomRates1D_T1.csv', 'w') as f:
        f.write("T1 [ms], Distribution, Cumulative \n")
        for i in range(len(T1)):
            f.write(f'{T1[i]:.6f}, {Snorm[i]:.6f}, {cumT1_1D[i]:.6f} \n')

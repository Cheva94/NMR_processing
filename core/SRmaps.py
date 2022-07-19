import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from scipy.signal import find_peaks
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d

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

def SRCPMG_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()
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

def Norm(Z, RGnorm, nH):
    norm = 1 / ((6.32589E-4 * np.exp(RGnorm/9) - 0.0854) * nH)
    return Z * norm

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

        if iter % 1000 == 0:
            TikhTerm = alpha * np.linalg.norm(S)**2
            ObjFunc = ZZT - 2 * np.trace(S.T @ K1TZK2) + np.trace(S.T @ K1TK1 @ S @ K2TK2) + TikhTerm

            Res = np.abs(ObjFunc - lastRes) / ObjFunc
            lastRes = ObjFunc
            print(f'# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break
    return S

def fitMag(tau1, tau2, T1, T2, S):
    t1 = range(len(tau1))
    d1 = range(len(T1))
    S1 = np.sum(S, axis=1)
    M1 = []

    for i in t1:
        m1 = 0
        for j in d1:
            m1 += S1[j] * (1 - np.exp(-tau1[i] / T1[j]))
        M1.append(m1[0])

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

def plot(tau1, tau2, Z, T1, T2, S, M1, M2, Out, nLevel, T1min, T1max, T2min, T2max, RGnorm, alpha, Back, nH, niniT1, niniT2, Map):
    fig, axs = plt.subplots(2,4)

    fig.suptitle(f'ns = 4 | RG = {RGnorm} dB | nH = {nH:.6f} | BG = {Back} | Alpha = {alpha} | nini SR = {niniT1} | nini CPMG = {niniT2}', fontsize='large')

    # SR: experimental y ajustada
    axs[0,0].scatter(tau1, Z[:, 0], label='Exp', color='coral')
    axs[0,0].plot(tau1, M1, label='Fit', color='teal')
    axs[0,0].set_xlabel(r'$\tau_1$ [ms]')
    axs[0,0].set_ylabel('SR')
    axs[0,0].legend()

    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.scatter(tau1[0:5], Z[:, 0][0:5], color='coral')
    axins1.plot(tau1[0:5], M1[0:5], color='teal')

    # SR: residuos
    axs[1,0].scatter(tau1, M1-Z[:, 0], color = 'blue')
    axs[1,0].axhline(0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(-0.1*np.max(Z[:, 0]), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,0].set_xlabel(r'$\tau$1 [ms]')
    axs[1,0].set_ylabel('Res. SR')

    # CPMG/FID/FID-CPMG: experimental y ajustada
    axs[0,1].scatter(tau2, Z[-1, :], label='Exp', color='coral')
    axs[0,1].plot(tau2, M2, label='Fit', color='teal')
    axs[0,1].legend()

    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=5)
    axins2.scatter(tau2[0:5], Z[-1, :][0:5], color='coral')
    axins2.plot(tau2[0:5], M2[0:5], color='teal')

    # CPMG/FID/FID-CPMG: residuos
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
    T2 = T2[2:]
    projT2 = np.sum(S, axis=0)
    projT2 = projT2[2:] / np.max(projT2[2:])
    peaks2, _ = find_peaks(projT2, height=0.1, distance = 5)
    peaks2x, peaks2y = T2[peaks2], projT2[peaks2]

    cumT2 = np.cumsum(projT2)
    cumT2 /= cumT2[-1]

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
    S = S[4:-9, 2:]

    axs[1,2].plot([10.0**mini, 10.0**maxi], [10.0**mini, 10.0**maxi], color='black', ls='-', alpha=0.7, zorder=-2, label = r'$T_1$ = $T_2$')
    for i in range(len(peaks2x)):
        axs[1,2].axvline(x=peaks2x[i], color='k', ls=':', lw=4)
    for i in range(len(peaks1x)):
        axs[1,2].axhline(y=peaks1x[i], color='k', ls=':', lw=4)
    axs[1,2].contour(T2, T1, S, nLevel, cmap='rainbow')
    axs[1,2].set_ylabel(r'$T_1$ [ms]')
    axs[1,2].set_xlim(10.0**T2min, 10.0**T2max)
    axs[1,2].set_ylim(10.0**T1min, 10.0**T1max)
    axs[1,2].set_xscale('log')
    axs[1,2].set_yscale('log')
    axs[1,2].legend(loc='lower right')

    for k in range(5):
        axs[1,3].scatter(tau2, gaussian_filter1d(Z[k, :], sigma=50), label=f'{k+1}')
    axs[1,3].legend()

    if Map == 'fid':
        axs[0,1].set_xlabel(r'$\tau_2^*$ [ms]')
        axs[0,1].set_ylabel('FID')

        axs[1,3].set_xlabel(r'$\tau_2^*$ [ms]')
        axs[1,3].set_ylabel('FID')

        axs[1,1].set_xlabel(r'$\tau_2^*$ [ms]')
        axs[1,1].set_ylabel('Res. FID')

        axs[0,3].set_xlabel(r'$T_2^*$ [ms]')
        axs[0,3].set_ylabel(r'Distrib. $T_2^*$')

        ax.set_ylabel(r'Cumul. $T_2^*$')

        axs[1,2].set_xlabel(r'$T_2^*$ [ms]')
    elif Map == 'cpmg':
        axs[0,1].set_xlabel(r'$\tau_2$ [ms]')
        axs[0,1].set_ylabel('CPMG')

        axs[1,3].set_xlabel(r'$\tau_2$ [ms]')
        axs[1,3].set_ylabel('CPMG')

        axs[1,1].set_xlabel(r'$\tau_2$ [ms]')
        axs[1,1].set_ylabel('Res. CPMG')

        axs[0,3].set_xlabel(r'$T_2$ [ms]')
        axs[0,3].set_ylabel(r'Distrib. $T_2$')

        ax.set_ylabel(r'Cumul. $T_2$')

        axs[1,2].set_xlabel(r'$T_2$ [ms]')
    elif Map == 'fidcpmg':
        axs[0,1].set_xlabel(r'$\tau_2^* | \tau_2$ [ms]')
        axs[0,1].set_ylabel('FID-CPMG')

        axs[1,3].set_xlabel(r'$\tau_2^* | \tau_2$ [ms]')
        axs[1,3].set_ylabel('FID-CPMG')

        axs[1,1].set_xlabel(r'$\tau_2^* | \tau_2$ [ms]')
        axs[1,1].set_ylabel('Res. FID-CPMG')

        axs[0,3].set_xlabel(r'$T_2^* | T_2$ [ms]')
        axs[0,3].set_ylabel(r'Distrib. $T_2^* | T_2$')

        ax.set_ylabel(r'Cumul. $T_2^* | T_2$')

        axs[1,2].set_xlabel(r'$T_2^* | T_2$ [ms]')

    plt.savefig(f'{Out}')

    with open(f'{Out}-DomRates1D_T1.csv', 'w') as f:
        f.write("T1 [ms], Distribution, Cumulative \n")
        for i in range(len(T1)):
            f.write(f'{T1[i]:.6f}, {projT1[i]:.6f}, {cumT1[i]:.6f} \n')

    with open(f'{Out}-DomRates1D_T2.csv', 'w') as f:
        f.write("T2 [ms], Distribution, Cumulative \n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}, {projT2[i]:.6f}, {cumT2[i]:.6f} \n')

def newSS(newS):
    return pd.read_csv(newS, header = None).to_numpy()

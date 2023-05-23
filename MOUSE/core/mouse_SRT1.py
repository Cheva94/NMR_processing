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

def SR_file(File, T1min, T1max, nini):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''

    data = pd.read_csv(File, header = None, sep = '\t').to_numpy()
    tau = data[:, 0] # In ms
    nP = len(tau) - nini

    buildUp = data[:, 1]

    nBin = 150
    S0 = np.ones(nBin)
    T1 = np.logspace(T1min, T1max, nBin)
    K = np.zeros((nP, nBin))

    for i in range(nP):
        K[i, :] = 1 - np.exp(-tau[i] / T1)

    pAcq = pd.read_csv('acqu.par', header = None, sep=' = ')
    shEcho = float(pAcq.iloc[9, 1])
    tEcho = float(pAcq.iloc[10, 1])
    nEcho = float(pAcq.iloc[20, 1])
    topEcho = float(pAcq.iloc[21, 1])
    nS = float(pAcq.iloc[23, 1])
    pulse = float(pAcq.iloc[25, 1])
    RD = float(pAcq.iloc[26, 1])

    return S0, T1, tau[nini:], K, buildUp[nini:], nP, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD

def NLI_FISTA(K, Z, alpha, S):
    '''
    Inversión de Laplace
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
            print(f'# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break

    return S[:, 0]

def fitMag(tau, T1, S, nP):
    '''
    Ajuste del decaimiento a partir de la distribución de T1.
    '''

    t = range(nP)
    d = range(len(T1))
    M = []
    for i in t:
        m = 0
        for j in d:
            m += S[j] * (1 - np.exp(- tau[i] / T1[j]))
        M.append(m)

    return M

def r_square(x, y, f, popt):
    '''
    Coeficiente de Pearson.
    '''

    residuals = y - f(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return 1 - ss_res / ss_tot

################################################################################
######################## Monoexponential section
################################################################################

def exp_1(t, M0, T1):
    return M0 * np.exp(- t / T1)

def fit_1(t, buildUp):
    popt, pcov = curve_fit(exp_1, t, buildUp, bounds=(0, np.inf), p0=[70, 2000])
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, buildUp, exp_1, popt)

    return popt, perr, r2

################################################################################
######################## Biexponential section
################################################################################

def exp_2(t, M0_1, T1_1, M0_2, T1_2):
    return M0_1 * np.exp(- t / T1_1) + M0_2 * np.exp(- t / T1_2)

def fit_2(t, buildUp):
    popt, pcov = curve_fit(exp_2, t, buildUp, bounds=(0, np.inf), p0=[70, 2000, 30, 1000])
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, buildUp, exp_2, popt)

    return popt, perr, r2

################################################################################
######################## Triexponential section
################################################################################

def exp_3(t, M0_1, T1_1, M0_2, T1_2, M0_3, T1_3):
    return M0_1 * np.exp(- t / T1_1) + M0_2 * np.exp(- t / T1_2) + M0_3 * np.exp(- t / T1_3)

def fit_3(t, buildUp):
    popt, pcov = curve_fit(exp_3, t, buildUp, bounds=(0, np.inf), p0=[70, 2000, 30, 1000, 10, 200])
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, buildUp, exp_3, popt)

    return popt, perr, r2

def plot(tau, Z, MLaplace, T1, S, Out, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD, alpha, Back, cumT1, nini, T1min, T1max, dataFit):
    '''
    Grafica resultados.
    '''

    fig, axs = plt.subplots(2, 3, gridspec_kw={'height_ratios': [3,1]})
    fig.suptitle(rf'nS={nS:.0f}    |    RD = {RD/1000:.2f} s    |    pulse = {pulse} $\mu$s    |    tE = {tEcho:.1f} $\mu$s    |    Ecos = {nEcho:.0f}    |    topSum = {topEcho:.0f}    |    shift = {shEcho:.1f} $\mu$s', fontsize='large')

    # SR: experimental y ajustada
    axs[0,0].set_title(f'Se descartaron {nini:.0f} puntos al comienzo.', fontsize='large')
    axs[0,0].scatter(tau, Z, label='Experimento', color='coral')
    axs[0,0].plot(tau, MLaplace, label='Fit Laplace', color='teal')
    axs[0,0].set_xlabel(r'$\tau$ [s]')
    axs[0,0].set_ylabel('SR')
    axs[0,0].legend()
    axs[0,0].axhline(0, c = 'k', lw = 4, ls = ':', zorder=-2)

    # SR: experimental y ajustada (en semilog)
    axs[0,1].set_title(f'¿Background restado? {Back}', fontsize='large')
    axs[0,1].scatter(tau, Z, label='Experimento', color='coral')
    axs[0,1].plot(tau, MLaplace, label='Fit Laplace', color='teal')
    axs[0,1].set_yscale('log')
    axs[0,1].set_xlabel(r'$\tau$ [s]')
    axs[0,1].set_ylabel('log(SR)')
    axs[0,1].legend()

    # SR: residuos de Laplace
    residuals = MLaplace-Z
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)
    R2Laplace = 1 - ss_res / ss_tot
    axs[1,0].set_title(fr'Ajuste con Laplace: R$^2$ = {R2Laplace:.6f}')
    axs[1,0].scatter(tau, residuals, color = 'blue')
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = ':')
    axs[1,0].set_xlabel(r'$\tau$ [s]')
    axs[1,0].axhline(0.1*np.max(Z), c = 'red', lw = 6, ls = '-')
    axs[1,0].axhline(-0.1*np.max(Z), c = 'red', lw = 6, ls = '-')

    # Distribución de T1
    # T1 = T1[2:-2]
    Snorm = S / np.max(S)
    peaks, _ = find_peaks(Snorm, height=0.025, distance = 12)
    peaksx, peaksy = T1[peaks], Snorm[peaks]

    axs[0,2].set_title(rf'$\alpha$ = {alpha}')
    axs[0,2].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[0,2].plot(T1, Snorm, label = 'Distrib.', color = 'teal')
    for i in range(len(peaksx)):
            axs[0,2].plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, color='black')
            axs[0,2].annotate(f'{peaksx[i]:.2f}', xy = (peaksx[i], peaksy[i] + 0.07), fontsize=30, ha='center')
    axs[0,2].set_xlabel(r'$T_2$ [s]')
    axs[0,2].set_ylabel(r'Distrib. $T_2$')
    axs[0,2].set_xscale('log')
    axs[0,2].set_ylim(-0.02, 1.2)
    axs[0,2].set_xlim(10.0**T1min, 10.0**T1max)

    cumT1norm = cumT1 / cumT1[-1]
    ax = axs[0,2].twinx()
    ax.plot(T1, cumT1norm, label = 'Cumul.', color = 'coral')
    ax.set_ylim(-0.02, 1.2)
    ax.set_ylabel(r'Cumul. $T_2$')

    axs[1,1].axis('off')
    axs[1,2].axis('off')

    # axs[1,1].annotate('>>> Ajuste monoexponencial <<<', xy = (0.5, 1.00), fontsize=30, ha='center')
    # axs[1,1].annotate(f'{dataFit[0,0]} --> {dataFit[1,0]}', xy = (0.5, 0.85), fontsize=30, ha='center')
    # axs[1,1].annotate(f'{dataFit[0,3]}', xy = (0.5, 0.70), fontsize=30, ha='center')

    # axs[1,1].annotate('>>> Ajuste biexponencial <<<', xy = (0.5, 0.45), fontsize=30, ha='center')
    # axs[1,1].annotate(f'{dataFit[2,0]} --> {dataFit[3,0]}', xy = (0.5, 0.30), fontsize=30, ha='center')
    # axs[1,1].annotate(f'{dataFit[2,1]} --> {dataFit[3,1]}', xy = (0.5, 0.15), fontsize=30, ha='center')
    # axs[1,1].annotate(f'{dataFit[2,3]}', xy = (0.5, 0.00), fontsize=30, ha='center')

    # axs[1,2].annotate('>>> Ajuste triexponencial <<<', xy = (0.5, 1.00), fontsize=30, ha='center')
    # axs[1,2].annotate(f'{dataFit[4,0]} --> {dataFit[5,0]}', xy = (0.5, 0.85), fontsize=30, ha='center')
    # axs[1,2].annotate(f'{dataFit[4,1]} --> {dataFit[5,1]}', xy = (0.5, 0.70), fontsize=30, ha='center')
    # axs[1,2].annotate(f'{dataFit[4,2]} --> {dataFit[5,2]}', xy = (0.5, 0.55), fontsize=30, ha='center')
    # axs[1,2].annotate(f'{dataFit[4,3]}', xy = (0.5, 0.40), fontsize=30, ha='center')

    plt.savefig(f'{Out}')

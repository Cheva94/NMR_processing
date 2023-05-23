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

def SRmap_file(File, T1min, T1max, T2min, T2max, cropT1, cropT2, Map):
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

    cropT2new = int(cropT2 + nFID)

    Nx = Ny = 150
    S0 = np.ones((Nx, Ny))
    S01D = np.ones(Nx)
    T1 = np.logspace(T1min, T1max, Nx)
    T2 = np.logspace(T2min, T2max, Ny)

    tau1 = pd.read_csv(File.split('.txt')[0]+"_t1.dat", header = None, delim_whitespace = True).to_numpy()
    tau2 = pd.read_csv(File.split('.txt')[0]+"_t2.dat", header = None, delim_whitespace = True).to_numpy()
    N1, N2 = len(tau1), len(tau2)
    tau1 = tau1[cropT1:]
    tau2 = tau2[cropT2new:]

    K1 = 1 - np.exp(-tau1 / T1)
    K2 = np.exp(-tau2 / T2)

    K1DSR = np.zeros((len(tau1), Nx))
    for i in range(len(tau1)):
        K1DSR[i, :] = 1 - np.exp(-tau1[i] / T1)
        
    K1DCPMG = np.zeros((len(tau2), Ny))
    for i in range(len(tau2)):
        K1DCPMG[i, :] = np.exp(-tau2[i] / T2)

    return S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, cropT2new, S01D, K1DSR, K1DCPMG

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

def Norm(Z, RG, N1, N2, cropT1, cropT2new):
    '''
    Normalización por ganancia y creación de matriz.
    '''

    norm = 1 / (6.32589E-4 * np.exp(RG/9) - 0.0854)
    Z = np.reshape(Z*norm, (N1, N2))[cropT1:, cropT2new:]
    
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

    return np.array(M1), np.array(M2)

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

def fitMag_SR(tau1, T1, S_1D):
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

    return np.array(M)

def fitMag_CPMG(tau2, T2, S_1D):
    '''
    Ajuste del decaimiento a partir de la distribución de T2 de la Laplace 1D.
    '''

    t = range(len(tau2))
    d = range(len(T2))
    M = []
    for i in t:
        m = 0
        for j in d:
            m += S_1D[j] * np.exp(- tau2[i] / T2[j])
        M.append(m[0])

    return np.array(M)

def SR1D_exp1cte(tau1, T1, M0):
    out = M0 * (1 - np.exp(-tau1 / T1))
    return out.flatten()


def SR1D_fit1cte(tau1, Z, T1min, T1max):
    '''
    Ajusta la SR aislada.
    '''

    guess_T1 = 10**((T1max-T1min) / 2)
    guess_M0 = Z[-1]

    popt, pcov = curve_fit(SR1D_exp1cte, tau1, Z, p0=[guess_T1, guess_M0])
    perr = np.sqrt(np.diag(pcov))

    residuals = Z - SR1D_exp1cte(tau1, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return popt[0], popt[1], perr[0], perr[1], r2

def SR1D_exp2cte(tau1, T1, M0, y0):
    out = y0 - M0 * np.exp(-tau1 / T1)
    return out.flatten()


def SR1D_fit2cte(tau1, Z, T1min, T1max):
    '''
    Ajusta la SR aislada.
    '''

    guess_T1 = 10**((T1max-T1min) / 2)
    guess_M0 = Z[-1]

    popt, pcov = curve_fit(SR1D_exp2cte, tau1, Z, p0=[guess_T1, guess_M0, 0])
    perr = np.sqrt(np.diag(pcov))

    residuals = Z - SR1D_exp2cte(tau1, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return popt[0], popt[1], popt[2], perr[0], perr[1], perr[2], r2

def CPMG1D_exp1cte(tau2, T2, M0):
    out = M0 * np.exp(-tau2 / T2)
    return out.flatten()


def CPMG1D_fit1cte(tau2, Z, T2min, T2max):
    '''
    Ajusta la SR aislada.
    '''

    guess_T2 = 10**((T2max-T2min) / 2)
    guess_M0 = Z[-1]

    popt, pcov = curve_fit(CPMG1D_exp1cte, tau2, Z, p0=[guess_T2, guess_M0])
    perr = np.sqrt(np.diag(pcov))

    residuals = Z - CPMG1D_exp1cte(tau2, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return popt[0], popt[1], perr[0], perr[1], r2

def CPMG1D_exp2cte(tau2, T2, M0, y0):
    out = y0 + M0 * np.exp(-tau2 / T2)
    return out.flatten()


def CPMG1D_fit2cte(tau2, Z, T2min, T2max):
    '''
    Ajusta la SR aislada.
    '''

    guess_T2 = 10**((T2max-T2min) / 2)
    guess_M0 = Z[-1]

    popt, pcov = curve_fit(CPMG1D_exp2cte, tau2, Z, p0=[guess_T2, guess_M0, 0])
    perr = np.sqrt(np.diag(pcov))

    residuals = Z - CPMG1D_exp2cte(tau2, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return popt[0], popt[1], popt[2], perr[0], perr[1], perr[2], r2


def plot(tau1, tau2, SR, CPMG, T1, T2, SR_Lap, CPMG_Lap, Mfit_SR, Mfit_CPMG, SR1D_T1, SR1D_M0, SR1D_T1sd, SR1D_M0sd, SR1D_r2, SR1D_T1bis, SR1D_M0bis, SR1D_y0, SR1D_T1sdbis, SR1D_M0sdbis, SR1D_y0sd, SR1D_r2bis, T1min, T1max, Out, CPMG1D_T1, CPMG1D_M0, CPMG1D_T1sd, CPMG1D_M0sd, CPMG1D_r2, CPMG1D_T1bis, CPMG1D_M0bis, CPMG1D_y0, CPMG1D_T1sdbis, CPMG1D_M0sdbis, CPMG1D_y0sd, CPMG1D_r2bis, T2min, T2max):
    '''
    Grafica resultados.
    '''

    fig, axs = plt.subplots(2,4)
    # fig.suptitle(f'Saqué 1 punto de la CPMG; sin corregir offset; sin omitir puntos negativos de la SR', fontsize='large')
    # fig.suptitle(f'Saqué 1 punto de la CPMG; sin corregir offset; omitiendo 16 puntos negativos de la SR', fontsize='large')
    chau = 0
    fig.suptitle(f'Saqué 1 puntos de la CPMG; corrigiendo offset; sin omitir puntos negativos de la SR', fontsize='large')
    
    # SR: experimental y ajustada (Laplace 2D)
    axs[0,0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,0].scatter(tau1, SR, label='Medición', color='black')#, zorder=5)
    axs[0,0].plot(tau1, Mfit_SR, label='Laplace')#, color='teal')#, zorder=0)
    axs[0,0].plot(tau1, SR1D_exp1cte(tau1, SR1D_T1, SR1D_M0), label='Exp. M0')#, color='darkgreen', zorder=1)
    axs[0,0].plot(tau1, SR1D_exp2cte(tau1, SR1D_T1, SR1D_M0, SR1D_y0), label='Exp. M0 y0')#, color='darkgreen', zorder=1)
    axs[0,0].set_xlabel(r'$\tau_1$ [ms]')
    axs[0,0].set_ylabel('SR')
    axs[0,0].legend(loc='lower center')
    
    # Inset del comienzo de la SR:
    axinsSR = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axinsSR.axhline(y=0, color='k', ls=':', lw=4)
    axinsSR.scatter(tau1[0:(15-chau)], SR[0:(15-chau)], label='Medición', color='black')#, zorder=5)
    axinsSR.plot(tau1[0:(15-chau)], Mfit_SR[0:(15-chau)], label='Laplace')#, color='teal')#, zorder=0)
    axinsSR.plot(tau1[0:(15-chau)], SR1D_exp1cte(tau1, SR1D_T1, SR1D_M0)[0:(15-chau)], label='Exp. M0')#, color='darkgreen', zorder=1)
    axinsSR.plot(tau1[0:(15-chau)], SR1D_exp2cte(tau1, SR1D_T1, SR1D_M0, SR1D_y0)[0:(15-chau)], label='Exp. M0 y0')#, color='darkgreen', zorder=1)
    
    # Distribución de T1 con Laplace 1D
    SR_Lap = SR_Lap[4:-9]
    T1 = T1[4:-9]
    Snorm = SR_Lap / np.max(SR_Lap)
    peaks, _ = find_peaks(SR_Lap)
    peaksx, peaksy = T1[peaks], Snorm[peaks]

    # cumT1_1D = np.cumsum(Snorm)
    # cumT1_1D /= cumT1_1D[-1]

    residuals = Mfit_SR-SR
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((SR - np.mean(SR))**2)
    R2_1D = 1 - ss_res / ss_tot

    # axs[1,2].set_title(fr'Laplace 1D: R$^2$ = {R2_1D:.6f}', fontsize='large')
    axs[1,0].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[1,0].plot(T1, Snorm)#, label = 'Distrib.', color = 'teal')
    for i in range(len(peaksx)):
        if peaksy[i] > 0.1:
            axs[1,0].plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, color='black')
            axs[1,0].annotate(fr'{peaksx[i]:.2f} ms (R$^2$={R2_1D:.6f})', xy = (peaksx[i], peaksy[i] + 0.07), fontsize=30, ha='center')
    axs[1,0].set_xlabel(r'$T_1$ [ms]')
    axs[1,0].set_ylabel(r'Distrib. $T_1$')
    axs[1,0].set_xscale('log')
    axs[1,0].set_ylim(-0.02, 1.2)
    axs[1,0].set_xlim(10.0**T1min, 10.0**T1max)
    
    # Diferencias entre ajustes
    axs[0,1].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,1].plot(tau1, Mfit_SR-SR, label='Lap-SR')
    axs[0,1].plot(tau1, SR1D_exp1cte(tau1, SR1D_T1, SR1D_M0)-SR, label='M0-SR')
    axs[0,1].plot(tau1, SR1D_exp2cte(tau1, SR1D_T1, SR1D_M0, SR1D_y0)-SR, label='(M0 y0)-SR')    
    axs[0,1].legend()

    axs[1,1].axis('off')
    
    axs[1,1].annotate(r'>>> Ajuste monoexponencial con $M = M_0 (1 - exp{-\frac{tau1}{T1}})$ <<<', xy = (0.5, 1.00), fontsize=30, ha='center')
    axs[1,1].annotate(fr'M0 = ({SR1D_M0:.2f} $\pm$ {SR1D_M0sd:.2f})', xy = (0.5, 0.85), fontsize=30, ha='center')
    axs[1,1].annotate(fr'T1 = ({SR1D_T1:.2f} $\pm$ {SR1D_T1sd:.2f})', xy = (0.5, 0.70), fontsize=30, ha='center')
    axs[1,1].annotate(fr'R$^2$ = {SR1D_r2:.6f}', xy = (0.5, 0.55), fontsize=30, ha='center')
    
    axs[1,1].annotate(r'>>> Ajuste monoexponencial con $M =  y_0 - M_0  exp{-\frac{tau1}{T1}})$ <<<', xy = (0.5, 0.45), fontsize=30, ha='center')
    axs[1,1].annotate(fr'M0 = ({SR1D_M0bis:.2f} $\pm$ {SR1D_M0sdbis:.2f})  |  y0 = ({SR1D_y0:.2f} $\pm$ {SR1D_y0sd:.2f})', xy = (0.5, 0.30), fontsize=30, ha='center')
    axs[1,1].annotate(fr'T1 = ({SR1D_T1bis:.2f} $\pm$ {SR1D_T1sdbis:.2f})', xy = (0.5, 0.15), fontsize=30, ha='center')
    axs[1,1].annotate(fr'R$^2$ = {SR1D_r2bis:.6f}', xy = (0.5, 0), fontsize=30, ha='center')
    
    
    
    
    
    
    # CPMG: experimental y ajustada (Laplace 2D)
    axs[0,2].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,2].scatter(tau2, CPMG, label='Medición', color='black')#, zorder=5)
    axs[0,2].plot(tau2, Mfit_CPMG, label='Laplace')#, color='teal')#, zorder=0)
    axs[0,2].plot(tau2, CPMG1D_exp1cte(tau2, CPMG1D_T1, CPMG1D_M0), label='Exp. M0')#, color='darkgreen', zorder=1)
    axs[0,2].plot(tau2, CPMG1D_exp2cte(tau2, CPMG1D_T1, CPMG1D_M0, CPMG1D_y0), label='Exp. M0 y0')#, color='darkgreen', zorder=1)
    axs[0,2].set_xlabel(r'$\tau_2$ [ms]')
    axs[0,2].set_ylabel('CPMG')
    axs[0,2].legend(loc='upper center')
    
    # Inset del comienzo de la CPMG:
    axinsSR = inset_axes(axs[0,2], width="30%", height="30%", loc=5)
    axinsSR.scatter(tau2[0:30], CPMG[0:30], label='Medición', color='black')#, zorder=5)
    axinsSR.plot(tau2[0:30], Mfit_CPMG[0:30], label='Laplace')#, color='teal')#, zorder=0)
    axinsSR.plot(tau2[0:30], CPMG1D_exp1cte(tau2, CPMG1D_T1, CPMG1D_M0)[0:30], label='Exp. M0')#, color='darkgreen', zorder=1)
    axinsSR.plot(tau2[0:30], CPMG1D_exp2cte(tau2, CPMG1D_T1, CPMG1D_M0, CPMG1D_y0)[0:30], label='Exp. M0 y0')#, color='darkgreen', zorder=1)
    
    # Distribución de T2 con Laplace 1D
    CPMG_Lap = CPMG_Lap[2:-2]
    T2 = T2[2:-2]
    Snorm = CPMG_Lap / np.max(CPMG_Lap)
    peaks, _ = find_peaks(CPMG_Lap)
    peaksx, peaksy = T2[peaks], Snorm[peaks]

    # cumT1_1D = np.cumsum(Snorm)
    # cumT1_1D /= cumT1_1D[-1]

    residuals = Mfit_CPMG-CPMG
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((CPMG - np.mean(CPMG))**2)
    R2_1D = 1 - ss_res / ss_tot

    # axs[1,2].set_title(fr'Laplace 1D: R$^2$ = {R2_1D:.6f}', fontsize='large')
    axs[1,2].axhline(y=0.1, color='k', ls=':', lw=4)
    axs[1,2].plot(T2, Snorm)#, label = 'Distrib.', color = 'teal')
    for i in range(len(peaksx)):
        if peaksy[i] > 0.1:
            axs[1,2].plot(peaksx[i], peaksy[i] + 0.05, lw = 0, marker=11, color='black')
            axs[1,2].annotate(fr'{peaksx[i]:.2f} ms (R$^2$={R2_1D:.6f})', xy = (peaksx[i], peaksy[i] + 0.07), fontsize=30, ha='center')
    axs[1,2].set_xlabel(r'$T_2$ [ms]')
    axs[1,2].set_ylabel(r'Distrib. $T_2$')
    axs[1,2].set_xscale('log')
    axs[1,2].set_ylim(-0.02, 1.2)
    axs[1,2].set_xlim(10.0**T2min, 10.0**T2max)
    
    # Diferencias entre ajustes
    axs[0,3].axhline(y=0, color='k', ls=':', lw=4)
    axs[0,3].plot(tau2, Mfit_CPMG-CPMG, label='Lap-CPMG')
    axs[0,3].plot(tau2, CPMG1D_exp1cte(tau2, CPMG1D_T1, CPMG1D_M0)-CPMG, label='M0-CPMG')
    axs[0,3].plot(tau2, CPMG1D_exp2cte(tau2, CPMG1D_T1, CPMG1D_M0, CPMG1D_y0)-CPMG, label='(M0 y0)-CPMG')    
    axs[0,3].legend()

    axs[1,3].axis('off')
    
    axs[1,3].annotate(r'>>> Ajuste monoexponencial con $M = M_0 exp{-\frac{tau2}{T2}})$ <<<', xy = (0.5, 1.00), fontsize=30, ha='center')
    axs[1,3].annotate(fr'M0 = ({CPMG1D_M0:.2f} $\pm$ {CPMG1D_M0sd:.2f})', xy = (0.5, 0.85), fontsize=30, ha='center')
    axs[1,3].annotate(fr'T2 = ({CPMG1D_T1:.2f} $\pm$ {CPMG1D_T1sd:.2f})', xy = (0.5, 0.70), fontsize=30, ha='center')
    axs[1,3].annotate(fr'R$^2$ = {CPMG1D_r2:.6f}', xy = (0.5, 0.55), fontsize=30, ha='center')
    
    axs[1,3].annotate(r'>>> Ajuste monoexponencial con $M =  y_0 + M_0  exp{-\frac{tau2}{T2}})$ <<<', xy = (0.5, 0.45), fontsize=30, ha='center')
    axs[1,3].annotate(fr'M0 = ({CPMG1D_M0bis:.2f} $\pm$ {CPMG1D_M0sdbis:.2f})  |  y0 = ({CPMG1D_y0:.2f} $\pm$ {CPMG1D_y0sd:.2f})', xy = (0.5, 0.30), fontsize=30, ha='center')
    axs[1,3].annotate(fr'T2 = ({CPMG1D_T1bis:.2f} $\pm$ {CPMG1D_T1sdbis:.2f})', xy = (0.5, 0.15), fontsize=30, ha='center')
    axs[1,3].annotate(fr'R$^2$ = {CPMG1D_r2bis:.6f}', xy = (0.5, 0), fontsize=30, ha='center')
    
    
    plt.savefig(f'{Out}')
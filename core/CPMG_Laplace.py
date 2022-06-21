import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from scipy.signal import find_peaks
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['coral', 'teal', 'tab:orange', 'mediumseagreen'])

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
plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linestyle"] = '-'

def CPMG_file(File, T2min, T2max, niniT2):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()
    tau = data[:, 0] # In ms
    nP = len(tau) - niniT2

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
    nS, p90, att, RD, tEcho, nEcho = pAcq.iloc[0, 1], pAcq.iloc[2, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1], 2*pAcq.iloc[6, 1], pAcq.iloc[7, 1]

    return S0, T2, tau[niniT2:], K, decay[niniT2:], nS, p90, att, RD, tEcho, nEcho

def PhCorr(decay):
    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        decay_ph = decay * np.exp(1j * tita)
        initVal[i] = decay_ph[0].real

    decay = decay * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))

    return decay.real

def Norm(Z, RGnorm, nH):
    norm = 1 / ((6.32589E-4 * np.exp(RGnorm/9) - 0.0854) * nH)
    return Z * norm

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
            print(f'# It = {iter} >>> Residue = {Res:.6f}')

            if Res < 1E-5:
                break
    print(f'¡Inversión lista! ({iter} iteraciones)')

    return S[:, 0]

def fitMag(tau, T2, S):
    t = range(len(tau))
    d = range(len(T2))
    M = []
    for i in t:
        m = 0
        for j in d:
            m += S[j] * np.exp(- tau[i] / T2[j])
        M.append(m)

    return M

def plot(tau, Z, M, T2, S, Out, nS, RGnorm, p90, att, RD, alpha, tEcho, nEcho, Back, nH, cumT2, niniT2, T2min, T2max):
    fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3,1]})

    fig.suptitle(f'nS={nS} | RG = {RGnorm} dB | nH = {nH:.6f} | RD = {RD} s | p90 = {p90} us  | BG = {Back} | Atten = {att} dB | tE = {tEcho:.1f} ms | Ecos = {nEcho:.0f} ({tau[-1]:.1f} ms) | Alpha = {alpha} | nini = {niniT2}', fontsize='large')

    # CPMG: experimental y ajustada
    axs[0,0].plot(tau, Z, label='Exp')
    axs[0,0].plot(tau, M, label='Fit')
    axs[0,0].set_xlabel(r'$\tau$ [ms]')
    axs[0,0].set_ylabel('CPMG')
    axs[0,0].legend()

    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.plot(tau[0:30], Z[0:30])
    axins1.plot(tau[0:30], M[0:30])

    # CPMG: experimental y ajustada (en semilog)
    axs[1,0].semilogy(tau, Z, label='Exp')
    axs[1,0].semilogy(tau, M, label='Fit')
    axs[1,0].set_xlim(-10, 300)
    axs[1,0].set_ylim(bottom=10**-3)
    axs[1,0].set_xlabel(r'$\tau$ [ms]')
    axs[1,0].set_ylabel('log(CPMG)')
    axs[1,0].legend()

    # CPMG: residuos
    axs[1,1].plot(tau, M-Z, color = 'blue', label='Fit-Exp')
    axs[1,1].axhline(0, c = 'k', lw = 4, ls = '-')
    axs[1,1].set_xlabel(r'$\tau$ [ms]')
    axs[1,1].set_ylabel('Res. CPMG')

    S /= np.max(S)
    peaks, _ = find_peaks(S)
    peaksx, peaksy = T2[peaks], S[peaks]

    axs[0,1].plot(T2, S, label = 'Distrib.', color = 'teal')
    for i in range(len(peaksx)):
        if peaksy[i] > 0.1:
            axs[0,1].plot(peaksx, peaksy + 0.05, lw = 0, marker=11, color='black')
            axs[0,1].annotate(f'{peaksx[i]:.0f}', xy = (peaksx[i], peaksy[i] + 0.07), fontsize=30, ha='center')
    axs[0,1].set_xlabel(r'$T_2$ [ms]')
    axs[0,1].set_ylabel(r'Distrib. $T_2$')
    axs[0,1].set_xscale('log')
    axs[0,1].set_ylim(-0.02, 1.2)
    axs[0,1].set_xlim(10.0**T2min, 10.0**T2max)

    ax = axs[0,1].twinx()
    ax.plot(T2, cumT2, label = 'Cumul.', color = 'coral')
    # ref = cumT2[0]
    # for x in range(len(T2)):
    #     if cumT2[x] < 0.01:
    #         continue
    #     elif (cumT2[x] - ref) < 0.0001:
    #         ax.annotate(f'{100*cumT2[x]:.0f} %', xy = (T2[-1], cumT2[x]), fontsize=30, ha='right', color='coral')
    #     ref = cumT2[x]
    ax.set_ylim(-0.02, 1.2)
    ax.set_ylabel(r'Cumul. $T_2$')

    plt.savefig(f'{Out}')

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

def SRCPMG_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2):
    data = pd.read_csv(File, header = None, delim_whitespace = True, comment='#').to_numpy()#[:, 0]
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
    if RGnorm == None:
        Norm = 1 / nH
    else:
        Norm = 1 / ((6.32589E-4 * np.exp(RGnorm/9) - 0.0854) * nH)
    return Z * Norm

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

def plot(tau1, tau2, Z, T1, T2, S, M1, M2, Out, nLevel, T1min, T1max, T2min, T2max, RGnorm, alpha, Back, nH, niniT1, niniT2):
    fig, axs = plt.subplots(2,4)

    fig.suptitle(f'RG = {RGnorm} dB | nH = {nH:.6f} | BG = {Back} | Alpha = {alpha} | nini SR = {niniT1} | nini CPMG = {niniT2}', fontsize='large')

    axs[0,0].plot(tau1, Z[:, 0], label='Exp')
    axs[0,0].plot(tau1, M1, label='Fit')
    axs[0,0].set_xlabel(r'$\tau_1$ [ms]')
    axs[0,0].set_ylabel('SR')
    axs[0,0].legend()

    axins1 = inset_axes(axs[0,0], width="30%", height="30%", loc=5)
    axins1.plot(tau1[0:(20-niniT1)], Z[:, 0][0:(20-niniT1)])
    axins1.plot(tau1[0:(20-niniT1)], M1[0:(20-niniT1)])

    axs[1,0].plot(tau1, M1-Z[:, 0], color = 'blue')
    axs[1,0].axhline(0, c = 'k', lw = 4, ls = '-')
    axs[1,0].set_xlabel(r'$\tau$1 [ms]')
    axs[1,0].set_ylabel('Residual SR')

    axs[0,1].plot(tau2, Z[-1, :], label='Exp')
    axs[0,1].plot(tau2, M2, label='Fit')
    axs[0,1].set_xlabel(r'$\tau_2$ [ms]')
    axs[0,1].set_ylabel('CPMG')
    axs[0,1].legend()

    axins2 = inset_axes(axs[0,1], width="30%", height="30%", loc=5)
    axins2.plot(tau2[0:(20-niniT2)], Z[-1, :][0:(20-niniT2)])
    axins2.plot(tau2[0:(20-niniT2)], M2[0:(20-niniT2)])

    axs[1,1].plot(tau2, M2-Z[-1, :], color = 'blue')
    axs[1,1].axhline(0, c = 'k', lw = 4, ls = '-')
    axs[1,1].set_xlabel(r'$\tau$2 [ms]')
    axs[1,1].set_ylabel('Residual CPMG')

    projT1 = np.sum(S, axis=1)
    peaks1, _ = find_peaks(projT1, height=0.005)
    peaks1x, peaks1y = T1[peaks1], projT1[peaks1]

    topPeak1 = np.max(peaks1y)
    ymax1 = topPeak1 + 0.3

    cumT1 = np.cumsum(projT1[15:-15])
    cumT1 /= cumT1[-1]

    axs[0,2].plot(T1[15:-15], projT1[15:-15], label = 'Distrib.', color = 'teal')
    for i in range(len(peaks1x)):
        if peaks1y[i] > 0.1 * topPeak1:
            axs[0,2].plot(peaks1x[i], peaks1y[i] + 0.1, lw = 0, marker=11, color='black')
            axs[0,2].annotate(f'{peaks1x[i]:.0f}', xy = (peaks1x[i], peaks1y[i] + 0.2), fontsize=30, ha = 'center')
    axs[0,2].set_xlabel(r'$T_1$ [ms]')
    axs[0,2].set_xscale('log')
    axs[0,2].set_xlim(10.0**T1min, 10.0**T1max)
    axs[0,2].set_ylim(bottom=-0.05, top=ymax1)

    ax = axs[0,2].twinx()
    ax.plot(T1[15:-15], cumT1, label = 'Cumul.', color = 'coral')
    ref = cumT1[0]
    for x in range(len(cumT1)):
        if cumT1[x] < 0.01:
            continue
        elif (cumT1[x] - ref) < 0.0001:
            ax.annotate(f'{100*cumT1[x]:.0f} %', xy = (T1[15:-15][-1], cumT1[x]), fontsize=30, ha='right', color='coral')
        ref = cumT1[x]
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('Cumulative')

    # lines1, labels1 = axs[0,2].get_legend_handles_labels()
    # lines2, labels2 = ax.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, loc=5)

    projT2 = np.sum(S, axis=0)
    peaks2, _ = find_peaks(projT2, height=0.005)
    peaks2x, peaks2y = T2[peaks2], projT2[peaks2]

    topPeak2 = np.max(peaks2y)
    ymax2 = topPeak2 + 0.3

    cumT2 = np.cumsum(projT2[15:-15])
    cumT2 /= cumT2[-1]

    axs[0,3].plot(T2[15:-15], projT2[15:-15], label = 'Distrib.', color = 'teal')
    for i in range(len(peaks2x)):
        if peaks2y[i] > 0.1 * topPeak2:
            axs[0,3].plot(peaks2x[i], peaks2y[i] + 0.1, lw = 0, marker=11, color='black')
            axs[0,3].annotate(f'{peaks2x[i]:.0f}', xy = (peaks2x[i], peaks2y[i] + 0.2), fontsize=30, ha = 'center')
    axs[0,3].set_xlabel(r'$T_2$ [ms]')
    axs[0,3].set_xscale('log')
    axs[0,3].set_ylim(bottom=-0.05, top=ymax2)
    axs[0,3].set_xlim(10.0**T2min, 10.0**T2max)

    ax = axs[0,3].twinx()
    ax.plot(T2[15:-15], cumT2, label = 'Cumul.', color = 'coral')
    ref = cumT2[0]
    for x in range(len(cumT2)):
        if cumT2[x] < 0.01:
            continue
        elif (cumT2[x] - ref) < 0.0001:
            ax.annotate(f'{100*cumT2[x]:.0f} %', xy = (T2[15:-15][-1], cumT2[x]), fontsize=30, ha='right', color='coral')
        ref = cumT2[x]
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('Cumulative')

    # lines1, labels1 = axs[0,3].get_legend_handles_labels()
    # lines2, labels2 = ax.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, loc=5)

    axs[1,2].plot(-projT1[15:-15], T1[15:-15], label = 'Distrib.', color = 'teal')
    axs[1,2].yaxis.tick_right()
    axs[1,2].yaxis.set_label_position("right")
    for i in range(len(peaks1x)):
        if peaks1y[i] > 0.1 * topPeak1:
            axs[1,2].plot(-peaks1y[i] - 0.1, peaks1x[i], lw = 0, marker=5, color='black')
            axs[1,2].annotate(f'{peaks1x[i]:.0f}', xy = (-peaks1y[i] - 0.2, peaks1x[i]), fontsize=30, va = 'center')
    # axs[1,2].set_ylabel(r'$T_1$ [ms]')
    axs[1,2].set_ylim(10.0**T1min, 10.0**T1max)
    axs[1,2].set_yscale('log')
    axs[1,2].set_xlim(left=-ymax1, right=0.05)
    axs[1,2].xaxis.set_visible(False)

    mini = np.max([T1min, T2min])
    maxi = np.min([T1max, T2max])

    axs[1,3].plot([10.0**mini, 10.0**maxi], [10.0**mini, 10.0**maxi], color='black', ls='-', alpha=0.7, zorder=-2, label = r'$T_1$ = $T_2$')
    axs[1,3].contour(T2[15:-15], T1[15:-15], S[15:-15, 15:-15], nLevel, cmap='rainbow')
    # axs[1,3].set_xlabel(r'$T_2$ [ms]')
    axs[1,3].set_ylabel(r'$T_1$ [ms]')
    axs[1,3].xaxis.tick_top()
    axs[1,3].set_xlim(10.0**T2min, 10.0**T2max)
    axs[1,3].set_ylim(10.0**T1min, 10.0**T1max)
    axs[1,3].set_xscale('log')
    axs[1,3].set_yscale('log')
    axs[1,3].legend(loc='lower right')

    plt.savefig(f'{Out}')

    with open(f'{Out}-DomRates1D_T1.csv', 'w') as f:
        f.write("T1 [ms], Distribution, Cumulative \n")
        for i in range(len(T1[15:-15])):
            f.write(f'{T1[15:-15][i]:.6f}, {projT1[15:-15][i]:.6f}, {cumT1[i]:.6f} \n')

    with open(f'{Out}-DomRates1D_T2.csv', 'w') as f:
        f.write("T2 [ms], Distribution, Cumulative \n")
        for i in range(len(T2[15:-15])):
            f.write(f'{T2[15:-15][i]:.6f}, {projT2[15:-15][i]:.6f}, {cumT2[i]:.6f} \n')

def newSS(newS):
    return pd.read_csv(newS, header = None).to_numpy()

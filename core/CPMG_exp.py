import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.optimize import curve_fit

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5

plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def CPMG_file(File, nini):
    '''
    Lectura del archivo de la medición y sus parámetros.
    '''
    data = pd.read_csv(File, header = None, delim_whitespace = True).to_numpy()

    t = data[:, 0] # In ms

    Re = data[:, 1]
    Im = data[:, 2]
    decay = Re + Im * 1j # Complex signal

    pAcq = pd.read_csv(File.split(".txt")[0]+'_acqs.txt', header = None, sep='\t')
    nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho = pAcq.iloc[0, 1], pAcq.iloc[1, 1], pAcq.iloc[2, 1], pAcq.iloc[3, 1], pAcq.iloc[4, 1], pAcq.iloc[5, 1], pAcq.iloc[6, 1], pAcq.iloc[7, 1], pAcq.iloc[8, 1]

    return t[nini:], decay[nini:], nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho

def PhCorr(signal):
    '''
    Corrección de fase.
    '''

    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        signal_ph = signal * np.exp(1j * tita)
        initVal[i] = signal_ph[0].real

    signal = signal * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))
    return signal.real

def Norm(decay, RGnorm, RG, m):
    '''
    Normalización por ganancia.
    '''

    norm = 1 / (6.32589E-4 * np.exp(RGnorm/9) - 0.0854)
    return decay * Norm

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

def exp_1(t, M0, T2):
    return M0 * np.exp(- t / T2)

def fit_1(t, decay):
    popt, pcov = curve_fit(exp_1, t, decay, bounds=(0, np.inf), p0=[70, 2000])
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_1, popt)

    M0, T2 = popt[0], popt[1]
    M0_SD, T2_SD = perr[0], perr[1]

    return popt, r2, M0, T2, M0_SD, T2_SD

def plot_1(t, decay, popt, tEcho, Out, nS, RG, RGnorm, p90, att, RD, nEcho, r2, Back, m, M0, T2):
    t_seg = t * 0.001

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    fig.suptitle(f'nS={nS} | RG = {RG} dB ({RGnorm}) | RD = {RD} s | p90 = {p90} us | Atten = {att} dB \n Ecos = {nEcho} | tE = {tEcho:.1f} ms | R2 = {r2:.4f} | BG = {Back} | m = {m} \n Pares (M0; T2 [ms]) = ({M0:.4f} ; {T2:.4f})', fontsize='small')

    ax1.plot(t_seg, decay, label='data')
    ax1.plot(t_seg, exp_1(t, *popt), label='mono')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('M')
    ax1.legend()

    ax2.semilogy(t_seg, decay, label='data')
    ax2.semilogy(t_seg, exp_1(t, *popt), label='mono')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('log(M)')
    ax2.legend()

    plt.savefig(f'{Out}')

def out_1(t, decay, tEcho, Out, r2, M0, T2, M0_SD, T2_SD, nS, RG, RGnorm, p90, att, RD, nEcho, Back, m):

    with open(f'{Out}.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s], tEcho [ms], nEcho, Back, m [g] \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD}, {tEcho:.1f}, {nEcho}, {Back}, {m} \n\n')

        f.write("M0, M0-SD, T2 [ms], T2-SD [ms], R2, tEcho [ms] \n")
        f.write(f'{M0:.4f}, {M0_SD:.4f}, {T2:.4f}, {T2_SD:.4f}, {r2:.4f} \n\n')

        f.write("t [ms], decay \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {decay[i]:.4f} \n')

################################################################################
######################## Biexponential section
################################################################################

def exp_2(t, M0_1, T2_1, M0_2, T2_2):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2)

def fit_2(t, decay):
    popt, pcov = curve_fit(exp_2, t, decay, bounds=(0, np.inf), p0=[70, 2000, 30, 1000])
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_2, popt)

    M0_1, T2_1, M0_2, T2_2 = popt[0], popt[1], popt[2], popt[3]
    M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = perr[0], perr[1], perr[2], perr[3]

    return popt, r2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD

def plot_2(t, decay, popt, tEcho, Out, nS, RG, RGnorm, p90, att, RD, nEcho, r2, Back, m, M0_1, T2_1, M0_2, T2_2):

    t_seg = t * 0.001

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    fig.suptitle(f'nS={nS} | RG = {RG} dB ({RGnorm}) | RD = {RD} s | p90 = {p90} us | Atten = {att} dB \n Ecos = {nEcho} | tE = {tEcho:.1f} ms | R2 = {r2:.4f} | BG = {Back} | m = {m} \n Pares (M0; T2 [ms]) = ({M0_1:.4f} ; {T2_1:.4f}) | ({M0_2:.4f} ; {T2_2:.4f})', fontsize='small')

    ax1.plot(t_seg, decay, label='data')
    ax1.plot(t_seg, exp_2(t, *popt), label='bi')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('M')
    ax1.legend()

    ax2.semilogy(t_seg, decay, label='data')
    ax2.semilogy(t_seg, exp_2(t, *popt), label='bi')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('log(M)')
    ax2.legend()

    plt.savefig(f'{Out}')

def out_2(t, decay, tEcho, Out, r2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, nS, RG, RGnorm, p90, att, RD, nEcho, Back, m):

    with open(f'{Out}.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s], tEcho [ms], nEcho, Back, m [g] \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD}, {tEcho:.1f}, {nEcho}, {Back}, {m} \n\n')

        f.write("M0_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], M0_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], R2, tEcho [ms] \n")
        f.write(f'{M0_1:.4f}, {M0_1_SD:.4f}, {T2_1:.4f}, {T2_1_SD:.4f}, {M0_2:.4f}, {M0_2_SD:.4f}, {T2_2:.4f}, {T2_2_SD:.4f}, {r2:.4f} \n\n')

        f.write("t [ms], decay \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {decay[i]:.4f} \n')

################################################################################
######################## Triexponential section
################################################################################

def exp_3(t, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2) + M0_3 * np.exp(- t / T2_3)

def fit_3(t, decay):
    popt, pcov = curve_fit(exp_3, t, decay, bounds=(0, np.inf), p0=[70, 2000, 30, 1000, 10, 200])
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_3, popt)

    M0_1, T2_1, M0_2, T2_2, M0_3, T2_3 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = perr[0], perr[1], perr[2], perr[3], perr[4], perr[5]

    return popt, r2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD

def plot_3(t, decay, popt, tEcho, Out, nS, RG, RGnorm, p90, att, RD, nEcho, r2, Back, m, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3):

    t_seg = t * 0.001

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    fig.suptitle(f'nS={nS} | RG = {RG} dB ({RGnorm}) | RD = {RD} s | p90 = {p90} us | Atten = {att} dB \n Ecos = {nEcho} | tE = {tEcho:.1f} ms | R2 = {r2:.4f} | BG = {Back} | m = {m} \n Pares (M0; T2 [ms]) = ({M0_1:.4f} ; {T2_1:.4f}) | ({M0_2:.4f} ; {T2_2:.4f}) | ({M0_3:.4f} ; {T2_3:.4f})', fontsize='small')

    ax1.plot(t_seg, decay, label='data')
    ax1.plot(t_seg, exp_3(t, *popt), label='tri')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('M')
    ax1.legend()

    ax2.semilogy(t_seg, decay, label='data')
    ax2.semilogy(t_seg, exp_3(t, *popt), label='tri')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('log(M)')
    ax2.legend()

    plt.savefig(f'{Out}')

def out_3(t, decay, tEcho, Out, r2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD, nS, RG, RGnorm, p90, att, RD, nEcho, Back, m):

    with open(f'{Out}.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s], tEcho [ms], nEcho, Back, m [g] \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD}, {tEcho:.1f}, {nEcho}, {Back}, {m} \n\n')

        f.write("M0_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], M0_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], M0_3, M0_3-SD, T2_3 [ms], T2_3-SD [ms], R2, tEcho [ms] \n")
        f.write(f'{M0_1:.4f}, {M0_1_SD:.4f}, {T2_1:.4f}, {T2_1_SD:.4f}, {M0_2:.4f}, {M0_2_SD:.4f}, {T2_2:.4f}, {T2_2_SD:.4f}, {M0_3:.4f}, {M0_3_SD:.4f}, {T2_3:.4f}, {T2_3_SD:.4f}, {r2:.4f} \n\n')

        f.write("t [ms], decay \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {decay[i]:.4f} \n')

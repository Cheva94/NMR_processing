#!/usr/bin/python3.6

'''
    Description: core functions for CPMG.py.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.optimize import curve_fit

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 35

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange',
                                        'mediumseagreen', 'k', 'm', 'y'])

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

def userfile(F):
    '''
    Extracts data from the .txt input file given by the user.
    '''

    data = pd.read_csv(F, header = None, delim_whitespace = True).to_numpy()

    t = data[:, 0] # In ms
    nP = len(t) # Number of points

    Re = data[:, 1]
    Im = data[:, 2]
    decay = Re + Im * 1j # Complex signal

    acq = File.split('.txt')[0]+'-acqs'+'.txt'
    acq = pd.read_csv(acq, header = None, delim_whitespace = True)
    nS, RG, RD, tau, nEcho = acq.iloc[0, 1], acq.iloc[1, 1], acq.iloc[5, 1], acq.iloc[6, 1], acq.iloc[7, 1]

    return t, nP, decay, nS, RG, RD, 2*tau, nEcho

def phase_correction(decay):
    '''
    Returns decay with phase correction (maximizing real part).
    '''

    initVal = {}
    for i in range(360):
        tita = np.deg2rad(i)
        decay_ph = decay * np.exp(1j * tita)
        initVal[i] = decay_ph[0].real

    decay = decay * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))
    return decay.real

def r_square(x, y, f, popt):
    '''
    Determines the R^2 when fitting.
    '''

    residuals = y - f(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return 1 - ss_res / ss_tot

def chi_square(x, y_obs, f, popt):
    '''
    Determines the X^2 when fitting.
    '''
    y_exp = f(x, *popt)
    residuals = y_obs - y_exp

    return np.sum(residuals**2 / y_exp)

################################################################################
######################## Monoexponential section
################################################################################

def exp_1(t, M0, T2):
    return M0 * np.exp(- t / T2)

def fit_1(t, decay):
    '''
    Fits monoexponential.
    '''

    popt, pcov = curve_fit(exp_1, t, decay, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_1, popt)
    chi2 = chi_square(t, decay, exp_1, popt)

    M0, T2 = popt[0], popt[1]
    M0_SD, T2_SD = perr[0], perr[1]

    return popt, r2, chi2, M0, T2, M0_SD, T2_SD

def plot_1(t, decay, popt, tEcho, fileRoot):
    '''
    Creates plot for monoexponential.
    '''

    t_seg = t * 0.001

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

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

    fig.suptitle(fr'$T_E$={tEcho:.2f} ms')

    plt.savefig(f'{fileRoot}-exp1')

def out_1(t, decay, tEcho, fileRoot, r2, chi2, M0, T2, M0_SD, T2_SD):
    '''
    Generates one output file with phase corrected CPMG and another one with the fitting parameters in the monoexponential case.
    '''

    with open(f'{fileRoot}-PhCorr.csv', 'w') as f:
        f.write("t [ms], decay \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {decay[i]:.4f} \n')

    with open(f'{fileRoot}-exp1.csv', 'w') as f:
        f.write("M0, M0-SD, T2 [ms], T2-SD [ms], R2, Chi2, tEcho [ms] \n")
        f.write(f'{M0:.4f}, {M0_SD:.4f}, {T2:.4f}, {T2_SD:.4f}, {r2:.4f}, {chi2:.4f}, {tEcho:.4f}')

################################################################################
######################## Biexponential section
################################################################################

def exp_2(t, M0_1, T2_1, M0_2, T2_2):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2)

def fit_2(t, decay):
    '''
    Fits biexponential.
    '''

    popt, pcov = curve_fit(exp_2, t, decay, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_2, popt)
    chi2 = chi_square(t, decay, exp_2, popt)

    M0_1, T2_1, M0_2, T2_2 = popt[0], popt[1], popt[2], popt[3]
    M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = perr[0], perr[1], perr[2], perr[3]

    return popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD

def plot_2(t, decay, popt, tEcho, fileRoot):
    '''
    Creates plot for biexponential.
    '''

    t_seg = t * 0.001

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

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

    fig.suptitle(fr'$T_E$={tEcho:.2f} ms')

    plt.savefig(f'{fileRoot}-exp2')

def out_2(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD):
    '''
    Generates one output file with phase corrected CPMG and another one with the fitting parameters in the biexponential case.
    '''

    with open(f'{fileRoot}-PhCorr.csv', 'w') as f:
        f.write("t [ms], decay \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {decay[i]:.4f} \n')

    with open(f'{fileRoot}-exp2.csv', 'w') as f:
        f.write("M0_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], M0_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], R2, Chi2, tEcho [ms] \n")
        f.write(f'{M0_1:.4f}, {M0_1_SD:.4f}, {T2_1:.4f}, {T2_1_SD:.4f}, {M0_2:.4f}, {M0_2_SD:.4f}, {T2_2:.4f}, {T2_2_SD:.4f}, {r2:.4f}, {chi2:.4f}, {tEcho:.4f}')

################################################################################
######################## Triexponential section
################################################################################

def exp_3(t, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2) + M0_3 * np.exp(- t / T2_3)

def fit_3(t, decay):
    '''
    Fits triexponential.
    '''

    popt, pcov = curve_fit(exp_3, t, decay, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_3, popt)
    chi2 = chi_square(t, decay, exp_3, popt)

    M0_1, T2_1, M0_2, T2_2, M0_3, T2_3 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = perr[0], perr[1], perr[2], perr[3], perr[4], perr[5]

    return popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD

def plot_3(t, decay, popt, tEcho, fileRoot):
    '''
    Creates plot for triexponential.
    '''

    t_seg = t * 0.001

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

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

    fig.suptitle(fr'$T_E$={tEcho:.2f} ms')

    plt.savefig(f'{fileRoot}-exp3')

def out_3(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD):
    '''
    Generates one output file with phase corrected CPMG and another one with the fitting parameters in the triexponential case.
    '''

    with open(f'{fileRoot}-PhCorr.csv', 'w') as f:
        f.write("t [ms], decay \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {decay[i]:.4f} \n')

    with open(f'{fileRoot}-exp3.csv', 'w') as f:
        f.write("M0_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], M0_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], M0_3, M0_3-SD, T2_3 [ms], T2_3-SD [ms], R2, Chi2, tEcho [ms] \n")
        f.write(f'{M0_1:.4f}, {M0_1_SD:.4f}, {T2_1:.4f}, {T2_1_SD:.4f}, {M0_2:.4f}, {M0_2_SD:.4f}, {T2_2:.4f}, {T2_2_SD:.4f}, {M0_3:.4f}, {M0_3_SD:.4f}, {T2_3:.4f}, {T2_3_SD:.4f}, {r2:.4f}, {chi2:.4f}, {tEcho:.4f}')

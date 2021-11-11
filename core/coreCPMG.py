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
plt.rcParams["axes.linewidth"] = 3
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange',
                                        'mediumseagreen', 'k', 'm', 'y'])

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 3

plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def userfile(input_file):
    '''
    Process the txt input file given by the user.
    '''

    data = pd.read_csv(input_file, header = None, delim_whitespace = True).to_numpy()

    t = data[:, 0]

    Re = data[:, 1]
    Im = data[:, 2]
    decay = Re + Im * 1j # Complex signal

    tEcho = t[1] - t[0]
    return t, decay, tEcho

def phase_correction(decay):
    '''
    Returns decay with phase correction (maximize real part).
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
    Fit monoexponential.
    '''

    popt, pcov = curve_fit(exp_1, t, decay, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_1, popt)
    chi2 = chi_square(t, decay, exp_1, popt)

    M0, T2 = popt[0], popt[1]
    M0_SD, T2_SD = perr[0], perr[1]

    return popt, r2, chi2, M0, T2, M0_SD, T2_SD

def plot_1(t, decay, popt, tEcho, input_file):
    '''
    Creates plot for monoexponential.
    '''

    fig, ax = plt.subplots()

    ax.plot(t, decay, lw=3, label='data')
    ax.plot(t, exp_1(t, *popt), lw=2, label='mono')

    ax.text(0.02,0.02, fr'$T_E$={tEcho} ms', ha='left', va='bottom',
            transform=ax.transAxes, size='small') #medium

    ax.set_xlabel('t [ms]')
    ax.set_ylabel(r'$Echo_{top}$ (t)')

    ax.legend()

    plt.savefig(f'{input_file.split(".txt")[0]}-exp1')

def out_1(t, decay, tEcho, input_file):
    '''
    Fit monoexponential, save optimized parameters with statistics and plot.
    '''

    popt, r2, chi2, M0, T2, M0_SD, T2_SD = fit_1(t, decay)

    plot_1(t, decay, popt, tEcho, input_file)

    output_file = input_file.split('.txt')[0]
    with open(f'{output_file}_fit-exp1.csv', 'w') as f:
        f.write("M0, M0-SD, T2 [ms], T2-SD [ms], R2, Chi2 \n")
        f.write(f'{M0:.4f}, {M0_SD:.4f}, {T2:.4f}, {T2_SD:.4f}, {r2:.4f}, {chi2:.4f}')

################################################################################
######################## Biexponential section
################################################################################

def exp_2(t, M0_1, T2_1, M0_2, T2_2):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2)

def fit_2(t, decay):
    '''
    Fit biexponential.
    '''

    popt, pcov = curve_fit(exp_2, t, decay, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_2, popt)
    chi2 = chi_square(t, decay, exp_2, popt)

    M0_1, T2_1, M0_2, T2_2 = popt[0], popt[1], popt[2], popt[3]
    M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = perr[0], perr[1], perr[2], perr[3]

    return popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD

def plot_2(t, decay, popt, tEcho, input_file):
    '''
    Creates plot for biexponential.
    '''

    fig, ax = plt.subplots()

    ax.plot(t, decay, lw=3, label='data')
    ax.plot(t, exp_2(t, *popt), lw=2, label='bi')

    ax.text(0.02,0.02, fr'$T_E$={tEcho} ms', ha='left', va='bottom',
            transform=ax.transAxes, size='small') #medium

    ax.set_xlabel('t [ms]')
    ax.set_ylabel(r'$Echo_{top}$ (t)')

    ax.legend()

    plt.savefig(f'{input_file.split(".txt")[0]}-exp2')

def out_2(t, decay, tEcho, input_file):
    '''
    Fit biexponential, save optimized parameters with statistics and plot.
    '''

    popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = fit_2(t, decay)

    plot_2(t, decay, popt, tEcho, input_file)

    output_file = input_file.split('.txt')[0]
    with open(f'{output_file}_fit-exp2.csv', 'w') as f:
        f.write("M0_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], M0_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], R2, Chi2 \n")
        f.write(f'{M0_1:.4f}, {M0_1_SD:.4f}, {T2_1:.4f}, {T2_1_SD:.4f}, {M0_2:.4f}, {M0_2_SD:.4f}, {T2_2:.4f}, {T2_2_SD:.4f}, {r2:.4f}, {chi2:.4f}')

################################################################################
######################## Triexponential section
################################################################################

def exp_3(t, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3):
    return M0_1 * np.exp(- t / T2_1) + M0_2 * np.exp(- t / T2_2) + M0_3 * np.exp(- t / T2_3)

def fit_3(t, decay):
    '''
    Fit triexponential.
    '''

    popt, pcov = curve_fit(exp_3, t, decay, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))

    r2 = r_square(t, decay, exp_3, popt)
    chi2 = chi_square(t, decay, exp_3, popt)

    M0_1, T2_1, M0_2, T2_2, M0_3, T2_3 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = perr[0], perr[1], perr[2], perr[3], perr[4], perr[5]

    return popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD

def plot_3(t, decay, popt, tEcho, input_file):
    '''
    Creates plot for triexponential.
    '''

    fig, ax = plt.subplots()

    ax.plot(t, decay, lw=3, label='data')
    ax.plot(t, exp_3(t, *popt), lw=2, label='tri')

    ax.text(0.02,0.02, fr'$T_E$={tEcho} ms', ha='left', va='bottom',
            transform=ax.transAxes, size='small') #medium

    ax.set_xlabel('t [ms]')
    ax.set_ylabel(r'$Echo_{top}$ (t)')

    ax.legend()

    plt.savefig(f'{input_file.split(".txt")[0]}-exp3')

def out_3(t, decay, tEcho, input_file):
    '''
    Fit triexponential, save optimized parameters with statistics and plot.
    '''

    popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = fit_3(t, decay)

    plot_3(t, decay, popt, tEcho, input_file)

    output_file = input_file.split('.txt')[0]
    with open(f'{output_file}_fit-exp3.csv', 'w') as f:
        f.write("M0_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], M0_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], M0_3, M0_3-SD, T2_3 [ms], T2_3-SD [ms], R2, Chi2 \n")
        f.write(f'{M0_1:.4f}, {M0_1_SD:.4f}, {T2_1:.4f}, {T2_1_SD:.4f}, {M0_2:.4f}, {M0_2_SD:.4f}, {T2_2:.4f}, {T2_2_SD:.4f}, {M0_3:.4f}, {M0_3_SD:.4f}, {T2_3:.4f}, {T2_3_SD:.4f}, {r2:.4f}, {chi2:.4f}')

################################################################################
######################## Multi - choosing
################################################################################

def out_multi(t, decay, tEcho, input_file):
    '''
    Fits mono-, bi- and tri- exponential decay to choose best fit.
    Also save optimized parameters with statistics and plot.
    '''

    popt_mono, r2_mono, chi2_mono, M0_mono, T2_mono, M0_SD_mono, T2_SD_mono = fit_1(t, decay)

    popt_bi, r2_bi, chi2_bi, M0_1_bi, T2_1_bi, M0_2_bi, T2_2_bi, M0_1_SD_bi, T2_1_SD_bi, M0_2_SD_bi, T2_2_SD_bi = fit_2(t, decay)

    popt_tri, r2_tri, chi2_tri, M0_1_tri, T2_1_tri, M0_2_tri, T2_2_tri, M0_3_tri, T2_3_tri, M0_1_SD_tri, T2_1_SD_tri, M0_2_SD_tri, T2_2_SD_tri, M0_3_SD_tri, T2_3_SD_tri = fit_3(t, decay)

    fig, ax = plt.subplots()

    ax.plot(t, decay, lw=3, label='data')
    ax.plot(t, exp_1(t, *popt_mono), lw=2, label='mono')
    ax.plot(t, exp_2(t, *popt_bi), lw=2, label='bi')
    ax.plot(t, exp_3(t, *popt_tri), lw=2, label='tri')

    ax.text(0.02,0.02, fr'$T_E$={tEcho} ms', ha='left', va='bottom',
            transform=ax.transAxes, size='small') #medium

    ax.set_xlabel('t [ms]')
    ax.set_ylabel(r'$Echo_{top}$ (t)')

    ax.legend()

    plt.savefig(f'{input_file.split(".txt")[0]}-all')

    output_file = input_file.split('.txt')[0]
    with open(f'{output_file}_fit-all.csv', 'w') as f:
        f.write("Components, R2, Chi2, M0_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], M0_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], M0_3, M0_3-SD, T2_3 [ms], T2_3-SD [ms] \n")
        f.write(f'Mono, {r2_mono:.4f}, {chi2_mono:.4f}, {M0_mono:.4f}, {M0_SD_mono:.4f}, {T2_mono:.4f}, {T2_SD_mono:.4f} \n')
        f.write(f'Bi, {r2_bi:.4f}, {chi2_bi:.4f}, {M0_1_bi:.4f}, {M0_1_SD_bi:.4f}, {T2_1_bi:.4f}, {T2_1_SD_bi:.4f}, {M0_2_bi:.4f}, {M0_2_SD_bi:.4f}, {T2_2_bi:.4f}, {T2_2_SD_bi:.4f} \n')
        f.write(f'Tri, {r2_tri:.4f}, {chi2_tri:.4f}, {M0_1_tri:.4f}, {M0_1_SD_tri:.4f}, {T2_1_tri:.4f}, {T2_1_SD_tri:.4f}, {M0_2_tri:.4f}, {M0_2_SD_tri:.4f}, {T2_2_tri:.4f}, {T2_2_SD_tri:.4f}, {M0_3_tri:.4f}, {M0_3_SD_tri:.4f}, {T2_3_tri:.4f}, {T2_3_SD_tri:.4f}')

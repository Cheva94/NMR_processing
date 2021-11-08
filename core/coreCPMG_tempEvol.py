#!/usr/bin/python3.6

'''
    Description: core functions for CPMG_tempEvol.py.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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
plt.rcParams["legend.fontsize"] = 25
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def t_arrays(input_file):
    '''
    Array with experimental time (time between CPMG experiments) and decay time.
    '''

    tDecay = pd.read_csv(input_file, header = None, delim_whitespace = True).to_numpy()[:, 0]
    tEcho = tDecay[1] - tDecay[0]

    File = input_file.split('_')[0]
    hms = pd.read_csv(f'{File}.txt', header = None, delim_whitespace = True, index_col=0).iloc[:, [1,3,5]].to_numpy()
    tEvol = []

    for exp in range(len(hms)):
        t = hms[exp,0] * 60 + hms[exp,1] + hms[exp,2] / 60 # minutes
        tEvol.append(t)

    tEvol -= tEvol[0]

    return tEvol, tDecay, tEcho

def decay_phCorr(input_file):
    '''
    Returns decay with phase correction (maximize real part).
    '''

    data = pd.read_csv(input_file, header = None, delim_whitespace = True).to_numpy()

    Re = data[:, 1]
    Im = data[:, 2]
    decay = Re + Im * 1j # Complex signal

    initVal = {}

    for i in range(360):
        tita = np.deg2rad(i)
        decay_ph = decay * np.exp(1j * tita)
        initVal[i] = decay_ph[0].real

    decay = decay * np.exp(1j * np.deg2rad(max(initVal, key=initVal.get)))
    return decay.real

def div_ceil(a, b):
    return int(np.ceil((a + b - 1) / b))

def plot_decay(fileRoot):
    '''
    Plots decay in time.
    '''

    A = pd.read_csv(f'{fileRoot}_dataDecay.csv').to_numpy()
    exps = np.shape(A)[1]-1
    count = div_ceil(exps, 10)

    c = np.arange(count)
    norm = mpl.colors.Normalize(vmin=c[0], vmax=c[-1])
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.rainbow)
    cmap.set_array([])

    fig, ax = plt.subplots()

    for i in range(count):
        ax.plot(A[:, 0], A[:, (10 * i) + 1], lw=3, color=cmap.to_rgba(i))

    cbar = fig.colorbar(cmap, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Start', 'End'], fontsize=15)

    ax.set_xlabel('t [ms]')
    ax.set_ylabel(r'$Echo_{top}$')

    plt.savefig(f'{fileRoot}_dataDecay')

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

    M0, T2 = popt[0], popt[1]
    M0_SD, T2_SD = perr[0], perr[1]

    return M0, T2, M0_SD, T2_SD

def out_1(tEvol, tDecay, Files):
    '''
    Extracts all the information and puts it together in two .csv files.
    '''

    params = []
    count = 1
    dataDecay = pd.DataFrame(tDecay, columns=['t [ms]'])

    for F in Files:
        decay = decay_phCorr(F)
        dataDecay[f'Exp #{count}'] = decay
        M0, T2, M0_SD, T2_SD = fit_1(tDecay, decay)
        params.append([M0, M0_SD, T2, T2_SD])
        count += 1

    params = np.array(params)
    name = Files[0].split("_")[0]
    dataDecay.to_csv(f'{name}_dataDecay.csv', index=False)
    with open(f'{name}_dataEvol-exp1.csv', 'w') as f:
        f.write("t [min], MO, M0-SD, T2 [ms], T2-SD [ms] \n")
        for exp in range(len(Files)):
            f.write(f'{tEvol[exp]}, {params[exp,0]:.4f}, {params[exp,1]:.4f}, {params[exp,2]:.4f}, {params[exp,3]:.4f} \n')

def plot_param1(fileRoot):
    '''
    Plots evolution of parameters.
    '''

    A = pd.read_csv(f'{fileRoot}_dataEvol-exp1.csv').to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    ax1.errorbar(A[:, 0], A[:, 1], yerr = A[:, 2], capsize = 15, marker = 'o', ls = 'None', ms = 15)
    ax1.set_xlabel('t [min]')
    ax1.set_ylabel('M0')

    ax2.errorbar(A[:, 0], A[:, 3], yerr = A[:, 4], capsize = 15, marker = 'o', ls = 'None', ms = 15)
    ax2.set_xlabel('t [min]')
    ax2.set_ylabel('T2 [ms]')

    plt.savefig(f'{fileRoot}_dataEvol-exp1')

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

    M0s =  {'1':popt[0], '2':popt[2]}
    M0s = [key for key in {k: v for k, v in sorted(M0s.items(), key=lambda item: item[1], reverse=True)}.keys()]

    comp_1 = (popt[0], perr[0], popt[1], perr[1])
    comp_2 = (popt[2], perr[2], popt[3], perr[3])
    comps = {'comp1':comp_1, 'comp2':comp_2}

    M0_1, M0_1_SD, T2_1, T2_1_SD = comps[f'comp{M0s[0]}']
    M0_2, M0_2_SD, T2_2, T2_2_SD = comps[f'comp{M0s[1]}']

    return M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD

def out_2(tEvol, tDecay, Files):
    '''
    Extracts all the information and puts it together in two .csv files.
    '''

    params = []
    count = 1
    dataDecay = pd.DataFrame(tDecay, columns=['t [ms]'])

    for F in Files:
        decay = decay_phCorr(F)
        dataDecay[f'Exp #{count}'] = decay
        M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = fit_2(tDecay, decay)
        params.append([M0_1, M0_1_SD, T2_1, T2_1_SD, M0_2, M0_2_SD, T2_2, T2_2_SD])
        count += 1

    params = np.array(params)
    name = Files[0].split("_")[0]
    dataDecay.to_csv(f'{name}_dataDecay.csv', index=False)
    with open(f'{name}_dataEvol-exp2.csv', 'w') as f:
        f.write("t [min], MO_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], MO_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms] \n")
        for exp in range(len(Files)):
            f.write(f'{tEvol[exp]}, {params[exp,0]:.4f}, {params[exp,1]:.4f}, {params[exp,2]:.4f}, {params[exp,3]:.4f}, {params[exp,4]:.4f}, {params[exp,5]:.4f}, {params[exp,6]:.4f}, {params[exp,7]:.4f} \n')

def plot_param2(fileRoot):
    '''
    Plots evolution of parameters.
    '''

    A = pd.read_csv(f'{fileRoot}_dataEvol-exp2.csv').to_numpy()

    fig, axs = plt.subplots(2, 2, figsize=(25, 20))

    axs[0,0].errorbar(A[:, 0], A[:, 1], yerr = A[:, 2], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 1')
    axs[0,0].set_xlabel('t [min]')
    axs[0,0].set_ylabel('M0')
    axs[0,0].set_title('Comp.1')

    axs[0,1].errorbar(A[:, 0], A[:, 3], yerr = A[:, 4], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 1')
    axs[0,1].set_xlabel('t [min]')
    axs[0,1].set_ylabel('T2 [ms]')
    axs[0,1].set_title('Comp.1')

    axs[1,0].errorbar(A[:, 0], A[:, 5], yerr = A[:, 6], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 2', color='mediumseagreen')
    axs[1,0].set_xlabel('t [min]')
    axs[1,0].set_ylabel('M0')
    axs[1,0].set_title('Comp.2')

    axs[1,1].errorbar(A[:, 0], A[:, 7], yerr = A[:, 8], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 2', color='mediumseagreen')
    axs[1,1].set_xlabel('t [min]')
    axs[1,1].set_ylabel('T2 [ms]')
    axs[1,1].set_title('Comp.2')

    plt.savefig(f'{fileRoot}_dataEvol-exp2')

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

    M0s =  {'1':popt[0], '2':popt[2], '3':popt[4]}
    M0s = [key for key in {k: v for k, v in sorted(M0s.items(), key=lambda item: item[1], reverse=True)}.keys()]

    comp_1 = (popt[0], perr[0], popt[1], perr[1])
    comp_2 = (popt[2], perr[2], popt[3], perr[3])
    comp_3 = (popt[4], perr[4], popt[5], perr[5])
    comps = {'comp1':comp_1, 'comp2':comp_2, 'comp3':comp_3}

    M0_1, M0_1_SD, T2_1, T2_1_SD = comps[f'comp{M0s[0]}']
    M0_2, M0_2_SD, T2_2, T2_2_SD = comps[f'comp{M0s[1]}']
    M0_3, M0_3_SD, T2_3, T2_3_SD = comps[f'comp{M0s[2]}']

    return M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD

def out_3(tEvol, tDecay, Files):
    '''
    Extracts all the information and puts it together in two .csv files.
    '''

    params = []
    count = 1
    dataDecay = pd.DataFrame(tDecay, columns=['t [ms]'])

    for F in Files:
        decay = decay_phCorr(F)
        dataDecay[f'Exp #{count}'] = decay
        M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = fit_3(tDecay, decay)
        params.append([M0_1, M0_1_SD, T2_1, T2_1_SD, M0_2, M0_2_SD, T2_2, T2_2_SD, M0_3, M0_3_SD, T2_3, T2_3_SD])
        count += 1

    params = np.array(params)
    name = Files[0].split("_")[0]
    dataDecay.to_csv(f'{name}_dataDecay.csv', index=False)
    with open(f'{name}_dataEvol-exp3.csv', 'w') as f:
        f.write("t [min], MO_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], MO_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], MO_3, M0_3-SD, T2_3 [ms], T2_3-SD [ms] \n")
        for exp in range(len(Files)):
            f.write(f'{tEvol[exp]}, {params[exp,0]:.4f}, {params[exp,1]:.4f}, {params[exp,2]:.4f}, {params[exp,3]:.4f}, {params[exp,4]:.4f}, {params[exp,5]:.4f}, {params[exp,6]:.4f}, {params[exp,7]:.4f}, {params[exp,8]:.4f}, {params[exp,9]:.4f}, {params[exp,10]:.4f}, {params[exp,11]:.4f} \n')

def plot_param3(fileRoot):
    '''
    Plots evolution of parameters.
    '''

    A = pd.read_csv(f'{fileRoot}_dataEvol-exp3.csv').to_numpy()

    fig, axs = plt.subplots(3, 2, figsize=(25, 30))

    axs[0,0].errorbar(A[:, 0], A[:, 1], yerr = A[:, 2], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 1')
    axs[0,0].set_xlabel('t [min]')
    axs[0,0].set_ylabel('M0')
    axs[0,0].set_title('Comp.1')

    axs[0,1].errorbar(A[:, 0], A[:, 3], yerr = A[:, 4], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 1')
    axs[0,1].set_xlabel('t [min]')
    axs[0,1].set_ylabel('T2 [ms]')
    axs[0,1].set_title('Comp.1')

    axs[1,0].errorbar(A[:, 0], A[:, 5], yerr = A[:, 6], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 2', color='mediumseagreen')
    axs[1,0].set_xlabel('t [min]')
    axs[1,0].set_ylabel('M0')
    axs[1,0].set_title('Comp.2')

    axs[1,1].errorbar(A[:, 0], A[:, 7], yerr = A[:, 8], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 2', color='mediumseagreen')
    axs[1,1].set_xlabel('t [min]')
    axs[1,1].set_ylabel('T2 [ms]')
    axs[1,1].set_title('Comp.2')

    axs[2,0].errorbar(A[:, 0], A[:, 9], yerr = A[:, 10], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 3', color='k')
    axs[2,0].set_xlabel('t [min]')
    axs[2,0].set_ylabel('M0')
    axs[2,0].set_title('Comp.3')

    axs[2,1].errorbar(A[:, 0], A[:, 11], yerr = A[:, 12], capsize = 15, marker = 'o', ls = 'None', ms = 15, label='Comp. 3', color='k')
    axs[2,1].set_xlabel('t [min]')
    axs[2,1].set_ylabel('T2 [ms]')
    axs[2,1].set_title('Comp.3')
    
    plt.savefig(f'{fileRoot}_dataEvol-exp3')

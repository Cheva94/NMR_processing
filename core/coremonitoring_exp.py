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
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["axes.prop_cycle"] = cycler('color', ['tab:orange', 'mediumseagreen', 'k', 'm', 'y'])

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 5

plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.fontsize"] = 25
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def t_arrays(fileRoot, t_wait, nFiles):
    '''
    Array with experimental time tEvol (time between CPMG experiments) and decay time.
    '''

    tDecay = pd.read_csv(f'{fileRoot}_001.txt', header = None, delim_whitespace = True).to_numpy()[:, 0]

    tEvol = [i * t_wait for i in range(nFiles)]

    return tEvol, tDecay

def decay_phCorr(input):
    '''
    Returns decay with phase correction (maximizing real part).
    '''

    data = pd.read_csv(input, header = None, delim_whitespace = True).to_numpy()

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

def plot_decay(fileRoot, tEvol, t_wait):
    '''
    Plots one decay every hour of experiment, comparting evolution in time.
    '''

    A = pd.read_csv(f'{fileRoot}-evolPhCorr.csv').to_numpy()
    exps = np.shape(A)[1]-1
    hour = 60/t_wait
    count = [10*i for i in range(div_ceil(exps, hour))]

    norm = mpl.colors.Normalize(vmin=count[0], vmax=count[-1])
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.rainbow)
    cmap.set_array([])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    t_seg = A[:, 0] * 0.001
    for i in count:
        ax1.plot(t_seg, A[:, i+1], lw=3, color=cmap.to_rgba(i))
        ax2.semilogy(t_seg, A[:, i+1], lw=3, color=cmap.to_rgba(i))

    cbar = fig.colorbar(cmap, ticks=[count[0], count[-1]])
    cbar.ax.set_yticklabels([f't = 0 h', f't = {tEvol[-1]/60:.0f} h'], fontsize=15)

    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('M')

    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('log(M)')

    plt.savefig(f'{fileRoot}-evolPhCorr')

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

    M0, T2 = popt[0], popt[1]
    M0_SD, T2_SD = perr[0], perr[1]

    return M0, T2, M0_SD, T2_SD

def out_1(tEvol, tDecay, Files, fileRoot, nFiles):
    '''
    Extracts all the information and puts it together in two .csv files. Also plots evolution of fits (0, 25, 50, 75 and 100 %).
    '''

    params = []
    count = 1
    dataDecay = pd.DataFrame(tDecay, columns=['t [ms]'])
    evolFiles = [int(nFiles * i) for i in [0.25, 0.5, 0.75, 1]]
    evolCounter = 0
    t_seg = tDecay * 0.001

    for F in Files:
        decay = decay_phCorr(F)
        dataDecay[f'Exp #{count}'] = decay
        M0, T2, M0_SD, T2_SD = fit_1(tDecay, decay)
        params.append([M0, M0_SD, T2, T2_SD])
        if (count == 1) or any(count == x for x in evolFiles):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

            ax1.plot(t_seg, decay, label='data')
            ax1.plot(t_seg, exp_1(tDecay, M0, T2), label='mono')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('M')
            ax1.legend()

            ax2.semilogy(t_seg, decay, label='data')
            ax2.semilogy(t_seg, exp_1(tDecay, M0, T2), label='mono')
            ax2.set_xlabel('t [s]')
            ax2.set_ylabel('log(M)')
            ax2.legend()

            fig.suptitle(f'File {count}: Evolution = {tEvol[count-1]/60:.0f} h ; Progress = {int(evolCounter*100)} %')

            plt.savefig(f'{fileRoot}-evolExp1-{str(evolCounter).replace(".", ",")}')

            evolCounter += 0.25

        count += 1

    dataDecay.to_csv(f'{fileRoot}-evolPhCorr.csv', index=False)

    params = np.array(params)
    with open(f'{fileRoot}-evolExp1.csv', 'w') as f:
        f.write("t [min], MO, M0-SD, T2 [ms], T2-SD [ms] \n")
        for exp in range(len(Files)):
            f.write(f'{tEvol[exp]}, {params[exp,0]:.4f}, {params[exp,1]:.4f}, {params[exp,2]:.4f}, {params[exp,3]:.4f} \n')

def plot_param1(fileRoot):
    '''
    Plots evolution of parameters in the monoexponential case.
    '''

    A = pd.read_csv(f'{fileRoot}-evolExp1.csv').to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

    t_h = A[:, 0] / 60

    ax1.errorbar(t_h, A[:, 1], yerr = A[:, 2], capsize = 8, marker = 'o', ls = ':', ms = 8)
    ax1.set_xlabel('t [h]')
    ax1.set_ylabel('M0')
    ax1.set_title(f'Comp. 1')

    ax2.errorbar(t_h, A[:, 3], yerr = A[:, 4], capsize = 8, marker = 'o', ls = ':', ms = 8)
    ax2.set_xlabel('t [h]')
    ax2.set_ylabel('T2 [ms]')
    ax2.set_title(f'Comp. 1')

    plt.savefig(f'{fileRoot}-evolExp1')

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

    T2s =  {'1':popt[1], '2':popt[3]}
    T2s = [key for key in {k: v for k, v in sorted(T2s.items(), key=lambda item: item[1], reverse=True)}.keys()]

    comp_1 = (popt[0], perr[0], popt[1], perr[1])
    comp_2 = (popt[2], perr[2], popt[3], perr[3])
    comps = {'comp1':comp_1, 'comp2':comp_2}

    M0_1, M0_1_SD, T2_1, T2_1_SD = comps[f'comp{T2s[0]}']
    M0_2, M0_2_SD, T2_2, T2_2_SD = comps[f'comp{T2s[1]}']

    return M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD

def out_2(tEvol, tDecay, Files, fileRoot, nFiles):
    '''
    Extracts all the information and puts it together in two .csv files. Also plots evolution of fits (0, 25, 50, 75 and 100 %).
    '''

    params = []
    count = 1
    dataDecay = pd.DataFrame(tDecay, columns=['t [ms]'])
    evolFiles = [int(nFiles * i) for i in [0.25, 0.5, 0.75, 1]]
    evolCounter = 0
    t_seg = tDecay * 0.001

    for F in Files:
        decay = decay_phCorr(F)
        dataDecay[f'Exp #{count}'] = decay
        M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = fit_2(tDecay, decay)
        params.append([M0_1, M0_1_SD, T2_1, T2_1_SD, M0_2, M0_2_SD, T2_2, T2_2_SD])
        if (count == 1) or any(count == x for x in evolFiles):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

            ax1.plot(t_seg, decay, label='data')
            ax1.plot(t_seg, exp_2(tDecay, M0_1, T2_1, M0_2, T2_2), label='bi')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('M')
            ax1.legend()

            ax2.semilogy(t_seg, decay, label='data')
            ax2.semilogy(t_seg, exp_2(tDecay, M0_1, T2_1, M0_2, T2_2), label='bi')
            ax2.set_xlabel('t [s]')
            ax2.set_ylabel('log(M)')
            ax2.legend()

            fig.suptitle(f'File {count}: Evolution = {tEvol[count-1]/60:.0f} h ; Progress = {int(evolCounter*100)} %')

            plt.savefig(f'{fileRoot}-evolExp2-{str(evolCounter).replace(".", ",")}')

            evolCounter += 0.25

        count += 1

    dataDecay.to_csv(f'{fileRoot}-evolPhCorr.csv', index=False)

    params = np.array(params)
    with open(f'{fileRoot}-evolExp2.csv', 'w') as f:
        f.write("t [min], MO_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], MO_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms] \n")
        for exp in range(len(Files)):
            f.write(f'{tEvol[exp]}, {params[exp,0]:.4f}, {params[exp,1]:.4f}, {params[exp,2]:.4f}, {params[exp,3]:.4f}, {params[exp,4]:.4f}, {params[exp,5]:.4f}, {params[exp,6]:.4f}, {params[exp,7]:.4f} \n')

def plot_param2(fileRoot):
    '''
    Plots evolution of parameters in the biexponential case.
    '''

    A = pd.read_csv(f'{fileRoot}-evolExp2.csv').to_numpy()

    fig, axs = plt.subplots(2, 2, figsize=(25, 20))

    t_h = A[:, 0] / 60

    axs[0,0].errorbar(t_h, A[:, 1], yerr = A[:, 2], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 1')
    axs[0,0].set_xlabel('t [h]')
    axs[0,0].set_ylabel('M0')
    axs[0,0].set_title(f'Comp. 1')

    axs[0,1].errorbar(t_h, A[:, 3], yerr = A[:, 4], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 1')
    axs[0,1].set_xlabel('t [h]')
    axs[0,1].set_ylabel('T2 [ms]')
    axs[0,1].set_title(f'Comp. 1')

    axs[1,0].errorbar(t_h, A[:, 5], yerr = A[:, 6], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 2', color='mediumseagreen')
    axs[1,0].set_xlabel('t [h]')
    axs[1,0].set_ylabel('M0')
    axs[1,0].set_title(f'Comp. 2')

    axs[1,1].errorbar(t_h, A[:, 7], yerr = A[:, 8], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 2', color='mediumseagreen')
    axs[1,1].set_xlabel('t [h]')
    axs[1,1].set_ylabel('T2 [ms]')
    axs[1,1].set_title(f'Comp. 2')

    plt.savefig(f'{fileRoot}-evolExp2')

    B = A[:, 5] / A[:, 1]

    fig, ax = plt.subplots()

    ax.plot(t_h, B)
    ax.set_xlabel('t [h]')
    ax.set_ylabel('M0_2 / M0_1')

    plt.savefig(f'{fileRoot}-evolExp2-M0ratio')

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

    T2s =  {'1':popt[1], '2':popt[3], '3':popt[5]}
    T2s = [key for key in {k: v for k, v in sorted(T2s.items(), key=lambda item: item[1], reverse=True)}.keys()]

    comp_1 = (popt[0], perr[0], popt[1], perr[1])
    comp_2 = (popt[2], perr[2], popt[3], perr[3])
    comp_3 = (popt[4], perr[4], popt[5], perr[5])
    comps = {'comp1':comp_1, 'comp2':comp_2, 'comp3':comp_3}

    M0_1, M0_1_SD, T2_1, T2_1_SD = comps[f'comp{T2s[0]}']
    M0_2, M0_2_SD, T2_2, T2_2_SD = comps[f'comp{T2s[1]}']
    M0_3, M0_3_SD, T2_3, T2_3_SD = comps[f'comp{T2s[2]}']

    return M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD

def out_3(tEvol, tDecay, Files, fileRoot, nFiles):
    '''
    Extracts all the information and puts it together in two .csv files.
    Also plots evolution of fits (0, 25, 50, 75 and 100 %).
    '''

    params = []
    count = 1
    dataDecay = pd.DataFrame(tDecay, columns=['t [ms]'])
    evolFiles = [int(nFiles * i) for i in [0.25, 0.5, 0.75, 1]]
    evolCounter = 0
    t_seg = tDecay * 0.001

    for F in Files:
        decay = decay_phCorr(F)
        dataDecay[f'Exp #{count}'] = decay
        M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = fit_3(tDecay, decay)
        params.append([M0_1, M0_1_SD, T2_1, T2_1_SD, M0_2, M0_2_SD, T2_2, T2_2_SD, M0_3, M0_3_SD, T2_3, T2_3_SD])
        if (count == 1) or any(count == x for x in evolFiles):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

            ax1.plot(t_seg, decay, label='data')
            ax1.plot(t_seg, exp_3(tDecay, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3), label='tri')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('M')
            ax1.legend()

            ax2.semilogy(t_seg, decay, label='data')
            ax2.semilogy(t_seg, exp_3(tDecay, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3), label='tri')
            ax2.set_xlabel('t [s]')
            ax2.set_ylabel('log(M)')
            ax2.legend()

            fig.suptitle(f'File {count}: Evolution = {tEvol[count-1]/60:.0f} h ; Progress = {int(evolCounter*100)} %')

            plt.savefig(f'{fileRoot}-evolExp3-{str(evolCounter).replace(".", ",")}')

            evolCounter += 0.25

        count += 1

    params = np.array(params)
    dataDecay.to_csv(f'{fileRoot}-evolPhCorr.csv', index=False)
    with open(f'{fileRoot}-evolExp3.csv', 'w') as f:
        f.write("t [min], MO_1, M0_1-SD, T2_1 [ms], T2_1-SD [ms], MO_2, M0_2-SD, T2_2 [ms], T2_2-SD [ms], MO_3, M0_3-SD, T2_3 [ms], T2_3-SD [ms] \n")
        for exp in range(len(Files)):
            f.write(f'{tEvol[exp]}, {params[exp,0]:.4f}, {params[exp,1]:.4f}, {params[exp,2]:.4f}, {params[exp,3]:.4f}, {params[exp,4]:.4f}, {params[exp,5]:.4f}, {params[exp,6]:.4f}, {params[exp,7]:.4f}, {params[exp,8]:.4f}, {params[exp,9]:.4f}, {params[exp,10]:.4f}, {params[exp,11]:.4f} \n')

def plot_param3(fileRoot):
    '''
    Plots evolution of parameters in the triexponential case.
    '''

    A = pd.read_csv(f'{fileRoot}-evolExp3.csv').to_numpy()

    fig, axs = plt.subplots(3, 2, figsize=(25, 30))

    t_h = A[:, 0] / 60

    axs[0,0].errorbar(t_h, A[:, 1], yerr = A[:, 2], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 1')
    axs[0,0].set_xlabel('t [h]')
    axs[0,0].set_ylabel('M0')
    axs[0,0].set_title(f'Comp. 1')

    axs[0,1].errorbar(t_h, A[:, 3], yerr = A[:, 4], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 1')
    axs[0,1].set_xlabel('t [h]')
    axs[0,1].set_ylabel('T2 [ms]')
    axs[0,1].set_title(f'Comp. 1')

    axs[1,0].errorbar(t_h, A[:, 5], yerr = A[:, 6], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 2', color='mediumseagreen')
    axs[1,0].set_xlabel('t [h]')
    axs[1,0].set_ylabel('M0')
    axs[1,0].set_title(f'Comp. 2')

    axs[1,1].errorbar(t_h, A[:, 7], yerr = A[:, 8], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 2', color='mediumseagreen')
    axs[1,1].set_xlabel('t [h]')
    axs[1,1].set_ylabel('T2 [ms]')
    axs[1,1].set_title(f'Comp. 2')

    axs[2,0].errorbar(t_h, A[:, 9], yerr = A[:, 10], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 3', color='k')
    axs[2,0].set_xlabel('t [h]')
    axs[2,0].set_ylabel('M0')
    axs[2,0].set_title(f'Comp. 3')

    axs[2,1].errorbar(t_h, A[:, 11], yerr = A[:, 12], capsize = 8, marker = 'o', ls = ':', ms = 8, label='Comp. 3', color='k')
    axs[2,1].set_xlabel('t [h]')
    axs[2,1].set_ylabel('T2 [ms]')
    axs[2,1].set_title(f'Comp. 3')

    plt.savefig(f'{fileRoot}-evolExp3')

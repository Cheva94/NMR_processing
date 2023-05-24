#!/usr/bin/python3.10

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'

plt.rcParams["figure.autolayout"] = True

def damped_sine(t, offset, amplitude, decayRate, phShift, period):
    return offset + amplitude * np.exp(- t / decayRate) * np.sin(np.pi * (t - phShift) / period)

def fit(t, magnetization, p0):
    popt, pcov = curve_fit(damped_sine, t, magnetization, p0 = p0, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def main():

    F_array = args.inputs
    Out = F_array[0].split('/')[0] + '/Nutac_fit'
    p0 = args.InitVal

    t, M0, MPts = [], [], []

    for F in F_array:
        data = pd.read_csv(F, sep='\t').to_numpy()
        for k in range(np.shape(data)[0]):
            t.append(data[k, 0])
            M0.append(data[k, 1])
            MPts.append(data[k, 2])

    t = np.array(t)
    M0 = np.array(M0)
    MPts = np.array(MPts)
    S = np.argsort(t)

    t = t[S]
    M0 = M0[S]
    MPts = MPts[S]

    if p0 == None:
        p0 = [0, 80000, 20, 1, 8]

        fig, axs = plt.subplots(1, 2, figsize=(25, 10))

        # Plot del primer punto de la FID
        popt, _ = fit(t, M0, p0)
        axs[0].set_title('Primer punto de cada FID')
        axs[0].scatter(t, M0, label='data', zorder=2, color='coral')
        axs[0].plot(t, damped_sine(t, *popt), label='fit', color='teal', zorder=0)
        axs[0].axhline(y=0, color='k', ls=':', lw=4)
        axs[0].set_xlabel('t [us]')
        axs[0].set_ylabel('M')
        axs[0].legend()

        # Plot del promedio de los primeros puntos de la FID
        popt, _ = fit(t, MPts, p0)
        axs[1].set_title('Primeros 10 puntos de cada FID')
        axs[1].scatter(t, MPts, label='data', zorder=2, color='coral')
        axs[1].plot(t, damped_sine(t, *popt), label='fit', color='teal', zorder=0)
        axs[1].axhline(y=0, color='k', ls=':', lw=4)
        axs[1].set_xlabel('t [us]')
        axs[1].set_ylabel('M')
        axs[1].legend()

        plt.savefig(f'{Out}')
    
    else:
        fig, axs = plt.subplots(1, 2, figsize=(25, 10))

        # Plot del primer punto de la FID
        popt, _ = fit(t, M0, p0)
        axs[0].set_title('Primer punto de cada FID')
        axs[0].scatter(t, M0, label='data', zorder=2, color='coral')
        axs[0].plot(t, damped_sine(t, *popt), label='fit', color='teal', zorder=0)
        axs[0].axhline(y=0, color='k', ls=':', lw=4)
        axs[0].set_xlabel('t [us]')
        axs[0].set_ylabel('M')
        axs[0].legend()

        # Plot del promedio de los primeros puntos de la FID
        popt, _ = fit(t, MPts, p0)
        axs[1].set_title('Primeros 10 puntos de cada FID')
        axs[1].scatter(t, MPts, label='data', zorder=2, color='coral')
        axs[1].plot(t, damped_sine(t, *popt), label='fit', color='teal', zorder=0)
        axs[1].axhline(y=0, color='k', ls=':', lw=4)
        axs[1].set_xlabel('t [us]')
        axs[1].set_ylabel('M')
        axs[1].legend()

        plt.savefig(f'{Out}')

    with open(f'{Out}.csv', 'w') as f:
        f.write("offset, amplitude, decayRate [us], phShift, period \n")
        f.write(f'{popt[0]:.2f}, {popt[1]:.2f}, {popt[2]:.2f}, {popt[3]:.2f}, {popt[4]:.2f} \n\n')

        f.write("t [us]\tM0\tMPts\n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}\t{M0[i]:.4f}\t{MPts[i]:.4f}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', help = "Path to the nutation files.", nargs='+')
    parser.add_argument('-p0', '--InitVal', help = 'Initial guess for parameters: offset, amplitude, decayRate, phShift, period.', nargs = 5, type = float)
    args = parser.parse_args()
    main()
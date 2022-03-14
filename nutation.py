#!/usr/bin/python3.9
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cycler import cycler

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
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["legend.edgecolor"] = 'black'

plt.rcParams["figure.figsize"] = 12.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linestyle"] = '-'

def damped_sine(t, offset, amplitude, decayRate, phShift, period):
    return offset + amplitude * np.exp(- t / decayRate) * np.sin(np.pi * (t - phShift) / period)

def fit(t, magnetization, p0):
    popt, pcov = curve_fit(damped_sine, t, magnetization, p0 = p0, bounds=(0, np.inf))
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def main():
    F_array = args.input
    Out = args.output
    p0 = args.InitVal

    t, M = [], []

    for F in F_array:
        data = pd.read_csv(F, header = None, delim_whitespace = True).to_numpy()
        for k in range(np.shape(data)[0]):
            t.append(data[k, 0])
            M.append(data[k, 1])

    t = np.array(t)
    M = np.array(M)
    S = np.argsort(t)

    t = t[S]
    M = M[S]

    if p0 == None:
        p0 = [10, 70, 20, 1, 10]
        popt, perr = fit(t, M, p0)

        fig, ax = plt.subplots()

        ax.scatter(t, M, label='data', zorder=2)
        ax.plot(t, damped_sine(t, *popt), label='fit', color='mediumseagreen', zorder=0)
        ax.set_xlabel('t [us]')
        ax.set_ylabel('M')
        ax.legend()

        plt.savefig(f'{Out}')

    else:
        popt, perr = fit(t, M, p0)

        fig, ax = plt.subplots()

        ax.scatter(t, M, label='data', zorder=2)
        ax.plot(t, damped_sine(t, *popt), label='fit', color='mediumseagreen', zorder=0)
        ax.set_xlabel('t [us]')
        ax.set_ylabel('M')
        ax.legend()

        plt.savefig(f'{Out}')

    with open(f'{Out}.csv', 'w') as f:
        f.write("offset, amplitude, decayRate [us], phShift, period \n")
        f.write(f'{popt[0]:.2f}, {popt[1]:.2f}, {popt[2]:.2f}, {popt[3]:.2f}, {popt[4]:.2f} \n\n')

        f.write("t [us], M \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.4f}, {M[i]:.4f} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the nutation files.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-p0', '--InitVal', help = 'Initial guess for parameters: offset, amplitude, decayRate, phShift, period.', nargs = 5, type = float)

    args = parser.parse_args()

    main()

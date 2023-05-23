#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

plt.rcParams["figure.figsize"] = 25, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'

def main():

    FileArr = args.input
    Out = args.output
    Labels = args.labels

    nF = range(len(FileArr))

    fig, (ax1, ax2) = plt.subplots(1,2)

    for k in nF:
        data = pd.read_csv(FileArr[k], delim_whitespace = True).to_numpy()
        t = data[:, 0] # In ms
        Distrib = data[:, 1]
        Cumul = data[:, 2]

        ax1.plot(t, Distrib, label = Labels[k])
        ax1.set_xlabel(r'$T_2$ [ms]')
        ax1.set_xscale('log')
        ax1.set_ylabel(r'Distrib. $T_2$')
        ax1.legend(loc='upper left')

        ax2.plot(t, Cumul, label = Labels[k])
        ax2.set_xlabel(r'$T_2$ [ms]')
        ax2.set_xscale('log')
        ax2.set_ylabel(r'Distrib. $T_2$')
        ax2.legend(loc='upper left')

    plt.savefig(f'{Out}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-L', '--labels', nargs='+')

    args = parser.parse_args()

    main()

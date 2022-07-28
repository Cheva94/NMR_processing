#!/usr/bin/python3.10
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

plt.rcParams["figure.figsize"] = 37.5, 10
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'

def main():

    FileArr = args.input
    Out = args.output
    Labels = args.labels

    nF = range(len(FileArr))

    for k in nF:
        data = pd.read_csv(FileArr[k], header = None, delim_whitespace = True).to_numpy()
        Laplace = data[3:153, :3]
        T2 = Laplace[:, 0]
        Cumulative = Laplace[:, 2]

        Time = data[154:, :3]
        t = Time[:, 0]
        Decay = Time[:, 1]

        return t[nini:], Decay[nini:], T2, Cumulative



        t, Decay, T2, Cumulative = CPMG_file(F, nini)
        signalArr.append(Decay)
        cumulArr.append(Cumulative)

    signalArr = np.reshape(signalArr, (nF, len(t))).T
    cumulArr = np.reshape(cumulArr, (nF, len(T2))).T

    plot(t, signalArr, T2, cumulArr, nF, Out, Labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-L', '--labels', nargs='+')

    args = parser.parse_args()

    main()

#!/usr/bin/python3.10

import argparse
import pandas as pd
import numpy as np
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

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams["lines.linestyle"] = '-'

plt.rcParams["figure.autolayout"] = True

def main():

    SDQ = args.DQ
    Sref = args.ref
    Out = SDQ.split('.csv')[0] + '_norm'

    SDQ = pd.read_csv(f'{SDQ}', sep='\t').to_numpy()
    vd = SDQ[:, 0]
    SDQ = SDQ[:, 1:]
    Sref = pd.read_csv(f'{Sref}', sep='\t').to_numpy()[:, 1:]
    Ssigma = SDQ + Sref

    SnDQ = SDQ/Ssigma

    fig, axs = plt.subplots(1, 3, figsize=(37.5, 10))

    # Plot del primer punto de la FID
    axs[0].set_title('Primer punto de cada FID')
    axs[0].scatter(vd, SnDQ[:, 0], label=rf'Max = {vd[SnDQ[:, 0] == np.max(SnDQ[:, 0])][0]} $\mu$s', color='tab:blue')
    axs[0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0].set_xlabel(r't [$\mu$s]')
    axs[0].legend()

    # Plot del promedio de los primeros puntos de la FID
    axs[1].set_title('Primeros 10 puntos de cada FID')
    axs[1].scatter(vd, SnDQ[:, 1], label=rf'Max = {vd[SnDQ[:, 1] == np.max(SnDQ[:, 1])][0]} $\mu$s', color='tab:orange')
    axs[1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1].set_xlabel(r't [$\mu$s]')
    axs[1].legend()

    # Plot de las áreas de los espectros
    axs[2].set_title('Área de los picos de cada espectro')
    axs[2].scatter(vd, SnDQ[:, 3], label=rf'Max = {vd[SnDQ[:, 3] == np.max(SnDQ[:, 3])][0]} $\mu$s', color='tab:green')
    axs[2].axhline(y=0, color='k', ls=':', lw=4)
    axs[2].set_xlabel(r't [$\mu$s]')
    axs[2].legend()

    plt.savefig(f'{Out}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('DQ', help = "Path to the DQ coherences.")
    parser.add_argument('ref', help = "Path to the ref DQ coherences.")
    args = parser.parse_args()
    main()
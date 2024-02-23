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

# Colores
Verde = '#08a189' # = 08A189 = 8 161 137    (principal)
Naranja = '#fa6210' # = FA6210 = 250 98 16    (secundario)
Morado = '#7f03fc' # =  = 127 3 252   (terciario)
Amarillo = '#f7f307' # =  = 247 243 7   (cuaternario)
Gris = '#626262' # =  = 98 98 98 (alternativo)
Negro = '#000000' # =  = 0 0 0 (base)

def main():

    SDQ = args.DQ
    Sref = args.ref
    Out = 'Norm'

    acq = pd.read_csv('acq_param.csv').to_numpy()
    DQfilter = acq[0][0].split('\t')[2]
    DQfilterzFil = acq[1][0].split('\t')[2]
    evol = acq[5][0].split('\t')[2]
    zFilter = acq[6][0].split('\t')[2]

    SDQ = pd.read_csv(f'{SDQ}', sep='\t').to_numpy()
    vd = SDQ[:, 0]
    SDQ = SDQ[:, 1:]
    Sref = pd.read_csv(f'{Sref}', sep='\t').to_numpy()[:, 1:]
    Ssigma = SDQ + Sref

    SnDQ = SDQ/Ssigma

    fig, axs = plt.subplots(1, 3, figsize=(37.5, 10))
    
    if DQfilter == '0.000000':
        acqdq = rf'No filter used | Evol = {evol} s | z-Filter = {zFilter} s'
    else:
        acqdq = rf'DQ-filter = {DQfilter} s | DQ-fil (z-fil) = {DQfilterzFil} s | Evol = {evol} s | z-Filter = {zFilter} s'

    fig.suptitle(acqdq, fontsize='large')

    # DQ
    axs[0].set_title('DQ')
    axs[0].scatter(vd, SDQ[:, 1], label=rf'Max= ({vd[SDQ[:, 1] == np.max(SDQ[:, 1])][0]} $\mu$s)', color=Morado)
    axs[0].axhline(y=0, color='k', ls=':', lw=4)
    axs[0].set_xlabel(r't [$\mu$s]')
    axs[0].legend()

    # Ref
    axs[1].set_title('Ref')
    axs[1].scatter(vd, Sref[:, 1], label=rf'Max= ({vd[Sref[:, 1] == np.max(Sref[:, 1])][0]} $\mu$s)', color=Naranja)
    axs[1].axhline(y=0, color='k', ls=':', lw=4)
    axs[1].set_xlabel(r't [$\mu$s]')
    axs[1].legend()

    # Norm
    axs[2].set_title('Norm')
    axs[2].scatter(vd, SnDQ[:, 1], label=rf'Max= ({vd[SnDQ[:, 1] == np.max(SnDQ[:, 1])][0]} $\mu$s)', color=Verde)
    axs[2].axhline(y=0, color='k', ls=':', lw=4)
    axs[2].set_xlabel(r't [$\mu$s]')
    axs[2].legend()

    plt.savefig(f'{Out}')

    with open(f'{Out}.csv', 'w') as f:
        f.write("vd [us]\tFID00\tFID Pts\tPeak Area\n")
        for i in range(len(vd)):
            f.write(f'{vd[i]:.1f}\t{SnDQ[i, 0]:.6f}\t{SnDQ[i, 1]:.6f}\t{SnDQ[i, 3]:.6f}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('DQ', help = "Path to the DQ coherences.")
    parser.add_argument('ref', help = "Path to the ref DQ coherences.")
    args = parser.parse_args()
    main()
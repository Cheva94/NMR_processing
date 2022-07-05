#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
from core.meanFID import *

def main():

    FileArr = args.input
    Out = args.output
    Labels = args.labels
    nini = args.niniValues
    nH = args.protonMoles

    nF = range(len(FileArr))

    fig, axs = plt.subplots(1,2)
    axins = inset_axes(axs[0], width="30%", height="30%", loc=5)
    Min, Max = [], []

    for k in nF:
        data = pd.read_csv(FileArr[k], header = None, delim_whitespace = True, comment='#').to_numpy()
        t = data[:, 0] # In ms
        t = t[nini:]
        Re = data[:, 1]
        Im = data[:, 2]
        signal = Re + Im * 1j
        signal = signal[nini:]
        signal = PhCorrNorm(signal, nH)

        Max.append(np.max(signal.real))
        Min.append(signal.real[40])

        axs[0].plot(t, signal.real, label=Labels[k])
        axs[1].plot(t, signal.real/np.max(signal.real), label=Labels[k])
        axins.plot(t, signal.real)

    axins.set_xlim(0,0.1)
    axins.set_ylim(np.min(Min), np.max(Max))

    axs[0].set_xlabel('t [ms]')
    axs[0].set_ylabel('FID')
    axs[0].set_xlim(right=20)
    axs[0].legend()

    axs[1].set_xlabel('t [ms]')
    axs[1].set_ylabel('FID (norm)')
    axs[1].set_xlim(right=20)
    axs[1].legend()

    plt.savefig(f'{Out}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-nH', '--protonMoles', type = float, default = 1)
    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-L', '--labels', nargs='+')
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning the FID.", type = int, default=0)

    args = parser.parse_args()

    main()

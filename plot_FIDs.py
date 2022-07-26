#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
from core.meanFID import *
import scipy.fft as FT

plt.rcParams["figure.figsize"] = 37.5, 10
plt.rcParams["lines.linewidth"] = 4

def main():

    FileArr = args.input
    Out = args.output
    Labels = args.labels
    nini = args.niniValues
    nH = args.protonMoles

    nF = range(len(FileArr))

    fig, axs = plt.subplots(1,3)
    Min, Max = [], []

    for k in nF:
        data = pd.read_csv(FileArr[k], header = None, delim_whitespace = True, comment='#').to_numpy()
        t = data[:, 0] # In ms
        DW = t[1] - t[0]
        t = t[nini:]
        nP = len(t) # Number of points

        Re = data[:, 1]
        Im = data[:, 2]
        signal = Re + Im * 1j
        signal = signal[nini:]
        signal = PhCorrNorm(signal, nH)

        Max.append(np.max(signal.real))
        Min.append(signal.real[40])

        axs[0].plot(t, signal.real, label=Labels[k])
        axs[1].plot(t, signal.real/np.max(signal.real), label=Labels[k])

        # Preparación del espectro
        zf = FT.next_fast_len(2**5 * nP)
        freq = FT.fftshift(FT.fftfreq(zf, d=DW)) # Hz scale
        CS = freq / 20 # ppm for Minispec scale
        spec = np.flip(FT.fftshift(FT.fft(signal, n = zf)))
        mask = (CS>-0.05)&(CS<0.05)
        if k==0:
            max_peak = np.max(spec.real[mask])
        spec /= max_peak

        axs[2].plot(CS, spec.real, label=Labels[k])

    axs[0].set_xlabel('t [ms]')
    axs[0].set_ylabel('FID')
    axs[0].set_xlim(right=20)
    axs[0].legend()

    axs[1].set_xlabel('t [ms]')
    axs[1].set_ylabel('FID (norm)')
    axs[1].set_xlim(right=20)
    axs[1].legend()

    axs[2].set_xlim(-0.06, 0.06)
    axs[2].set_ylim(-0.05, 1.2)
    axs[2].set_xlabel(r'$\delta$ [ppm]')
    axs[2].axvline(x=0, color='k', ls=':', lw=2)
    axs[2].axhline(y=0, color='k', ls=':', lw=2)
    axs[2].legend()

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

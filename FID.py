#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
from core.coreFID import *

def main():

    File = args.input
    Out = args.output
    m = args.mass
    RGnorm = args.RGnorm
    show = args.ShowPlot
    Back = args.background

    if Back == None:
        t, signal, nP, DW, nS, RG, p90, att, RD = FID_file(File)
        signal = PhCorr(signal)

    else:
        t, signal, nP, DW, nS, RG, p90, att, RD = FID_file(File)
        signal = PhCorr(signal)

        _, Back, _, _, _, _, _, _, _ = FID_file(Back)
        Back = PhCorr(Back)

        Re = signal.real - Back.real
        Im = signal.imag - Back.imag

        signal = Re + Im * 1j

    signal = Norm(signal, RGnorm, RG, m)

    with open(f'{Out}.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s] \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD} \n\n')

        f.write("t [ms], Re[FID], Im[FID] \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.6f}, {signal.real[i]:.6f}, {signal.imag[i]:.6f} \n')

    plot(t, signal, nP, DW, nS, RGnorm, RG, p90, att, RD, Out)

    if show == 'on':
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-m', '--mass', help = "Sample mass.", type = float, default = 1)
    parser.add_argument('-RGnorm', '--RGnorm', help = "Normalize by RG.", default = "off")
    parser.add_argument('-show', '--ShowPlot', help = "Show plots.", default = 'off')
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

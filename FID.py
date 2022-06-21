#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: February, 2022.
'''

import argparse
from core.FID import *

def main():

    File = args.input
    Out = args.output
    nH = args.protonMoles
    RGnorm = args.RGnorm
    Back = args.background
    nini = args.niniValues

    if Back == None:
        t, signal, nP, DW, nS, RG, p90, att, RD = FID_file(File, nini)
        signal = PhCorr(signal)

    else:
        t, signal, nP, DW, nS, RG, p90, att, RD = FID_file(File, nini)
        signal = PhCorr(signal)

        _, back, _, _, _, _, _, _, _ = FID_file(Back, nini)
        back = PhCorr(back)

        Re = signal.real - back.real
        Im = signal.imag - back.imag

        signal = Re + Im * 1j

    signal = Norm(signal, RGnorm, RG, nH)

    if Back != None:
        Back = "Yes"

    with open(f'{Out}_FIDandParams.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s], Back, nH [mol], nini \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD}, {Back}, {nH}, {nini} \n\n')

        f.write("t [ms], Re[FID]/molH, Im[FID]/molH \n")
        for i in range(len(t)):
            f.write(f'{t[i]:.6f}, {signal.real[i]:.6f}, {signal.imag[i]:.6f} \n')

    plot(t, signal, nP, DW, nS, RGnorm, RG, p90, att, RD, Out, Back, nH, nini)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-nH', '--protonMoles', type = float, default = 1)
    parser.add_argument('-RGnorm', '--RGnorm', help = "Normalize by RG. Default: on", default = "on")
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=0)

    args = parser.parse_args()

    main()

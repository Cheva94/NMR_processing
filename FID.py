#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
from core.FID import *

def main():

    File = args.input
    Out = args.output
    Back = args.background
    nini = args.croppedValues

    print('Processing...')

    if Back == None:
        t, signal, nP, DW, nS, RDT, RG, att, RD, p90 = FID_file(File, nini)
        signal = PhCorr(signal)

        Back = "Nein!"

    else:
        t, signal, nP, DW, nS, RDT, RG, att, RD, p90 = FID_file(File, nini)
        signal = PhCorr(signal)

        _, back, _, _, _, _, _, _ = FID_file(Back, nini)
        back = PhCorr(back)

        Re = signal.real - back.real
        Im = signal.imag - back.imag

        signal = Re + Im * 1j

        Back = "Ja!"

    signal = Norm(signal, RG)

    print('Plotting...')

    plot(t, signal, nP, DW, nS, RDT, RG, att, RD, p90, Out, Back, nini)

    print('Writing output...')

    with open(f'{Out}.csv', 'w') as f:
        f.write("t [ms]\tRe[FID]\tIm[FID] \n")
        for i in range(nP):
            f.write(f'{t[i]:.6f}\t{signal.real[i]:.6f}\t{signal.imag[i]:.6f} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the FID file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning the FID.", type = int, default=0)

    args = parser.parse_args()

    main()

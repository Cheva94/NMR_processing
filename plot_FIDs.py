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

    signalArr = []
    nF = len(FileArr)

    for F in FileArr:
        t, signal = FID_file(F, nini)
        signal = PhCorrNorm(signal, nH)
        signalArr.append(signal)

    signalArr = np.reshape(signalArr, (nF, len(t))).T

    plot(t, signalArr, nF, Out, Labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-nH', '--protonMoles', type = float, default = 1)
    parser.add_argument('input', help = "Path to the FID file.", nargs = '+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-L', '--labels', nargs='+')
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning the FID.", type = int, default=0)

    args = parser.parse_args()

    main()

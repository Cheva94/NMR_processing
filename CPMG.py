#!/usr/bin/python3.8

'''
    Description: fit exponential decay (1, 2 or 3 components) with CPMG data.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreCPMG import *

def main():

    Files = args.input_file

    if args.monoexponential:
        for F in Files:
            t, decay, tEcho = userfile(F)
            decay = phase_correction(decay)
            out_1(t, decay, tEcho, F)
    elif args.biexponential:
        for F in Files:
            t, decay, tEcho = userfile(F)
            decay = phase_correction(decay)
            out_2(t, decay, tEcho, F)
    elif args.triexponential:
        for F in Files:
            t, decay, tEcho = userfile(F)
            decay = phase_correction(decay)
            out_3(t, decay, tEcho, F)
    elif args.multiexponential:
        for F in Files:
            t, decay, tEcho = userfile(F)
            decay = phase_correction(decay)
            out_multi(t, decay, tEcho, F)
    else:
        print('Must choose an option: -exp1, -exp2, -exp3 or -all. Use -h for guidance.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', help = "Path to the inputs file.",
                        nargs = '+')

    parser.add_argument('-all', '--multiexponential', action = 'store_true',
                        help = "Fits mono-, bi- and tri- exponential decay to \
                        choose best fit.")

    parser.add_argument('-exp1', '--monoexponential', action = 'store_true',
                        help = "Fits monoexponential decay.")

    parser.add_argument('-exp2', '--biexponential', action = 'store_true',
                        help = "Fits biexponential decay.")

    parser.add_argument('-exp3', '--triexponential', action = 'store_true',
                        help = "Fits triexponential decay.")

    args = parser.parse_args()

    main()

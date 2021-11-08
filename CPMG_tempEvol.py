#!/usr/bin/python3.6

'''
    Description: fits exponential decay (1, 2 or 3 components) with CPMG data,
                considering temporal evolution of parameters.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreCPMG_tempEvol import *

def main():

    Files = args.input_file

    if args.monoexponential:
        tEvol, tDecay, tEcho = t_arrays(Files[0])
        out_1(tEvol, tDecay, Files)
        plot_1(Files[0])

    # elif args.biexponential:
    #     for F in Files:
    #         t, decay, tEcho = userfile(F)
    #         decay = phase_correction(decay)
    #         out_2(t, decay, tEcho, F)
    # elif args.triexponential:
    #     for F in Files:
    #         t, decay, tEcho = userfile(F)
    #         decay = phase_correction(decay)
    #         out_3(t, decay, tEcho, F)
    else:
        print('Must choose an option: -exp1, -exp2 or -exp3. Use -h for guidance.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', help = "Path to the inputs file.",
                        nargs = '+')

    parser.add_argument('-exp1', '--monoexponential', action = 'store_true',
                        help = "Fits monoexponential decay.")

    parser.add_argument('-exp2', '--biexponential', action = 'store_true',
                        help = "Fits biexponential decay.")

    parser.add_argument('-exp3', '--triexponential', action = 'store_true',
                        help = "Fits triexponential decay.")

    args = parser.parse_args()

    main()

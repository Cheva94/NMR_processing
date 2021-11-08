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
        fileRoot = Files[0].split('_')[0]
        plot_decay(fileRoot)
        plot_param1(fileRoot)
    elif args.biexponential:
        tEvol, tDecay, tEcho = t_arrays(Files[0])
        out_2(tEvol, tDecay, Files)
        fileRoot = Files[0].split('_')[0]
        plot_decay(fileRoot)
        plot_param2(fileRoot)
    elif args.triexponential:
        tEvol, tDecay, tEcho = t_arrays(Files[0])
        out_3(tEvol, tDecay, Files)
        fileRoot = Files[0].split('_')[0]
        plot_decay(fileRoot)
        plot_param3(fileRoot)
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

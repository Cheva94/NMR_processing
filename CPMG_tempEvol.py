#!/usr/bin/python3.6

'''
    Description: corrects phase of CPMG decay and fits it considering 1, 2 or 3 exponentials, considering temporal evolution of parameters. Then plots one decay per hour in semilog scale. Also one fit for quarter of experiment progress. All the processed data will be also saved in ouput files (.csv).
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreCPMG_tempEvol import *

def main():

    Files = args.input
    nFiles = len(Files)
    fileRoot = Files[0].split('_0')[0]

    t_wait = args.t_wait
    tEvol, tDecay = t_arrays(fileRoot, t_wait, nFiles)

    if args.monoexponential:
        out_1(tEvol, tDecay, Files, fileRoot, nFiles)
        plot_decay(fileRoot, tEvol, t_wait)
        plot_param1(fileRoot)
    elif args.biexponential:
        out_2(tEvol, tDecay, Files, fileRoot, nFiles)
        plot_decay(fileRoot, tEvol, t_wait)
        plot_param2(fileRoot)
    elif args.triexponential:
        out_3(tEvol, tDecay, Files, fileRoot, nFiles)
        plot_decay(fileRoot, tEvol, t_wait)
        plot_param3(fileRoot)
    else:
        print('Must choose an option: -exp1, -exp2 or -exp3. Use -h for guidance.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Corrects phase of CPMG decay and fits it considering 1, 2 or 3 exponentials, considering temporal evolution of parameters. Then plots one decay per hour in semilog scale. Also one fit for quarter of experiment progress. All the processed data will be also saved in ouput files (.csv).')

    parser.add_argument('input', help = "Path to the inputs file.",
                        nargs = '+')

    parser.add_argument('t_wait', help = "Waiting time (in minutes) between CPMG experiments.", type=int)

    parser.add_argument('-exp1', '--monoexponential', action = 'store_true',
                        help = "Fits monoexponential decay.")

    parser.add_argument('-exp2', '--biexponential', action = 'store_true',
                        help = "Fits biexponential decay.")

    parser.add_argument('-exp3', '--triexponential', action = 'store_true',
                        help = "Fits triexponential decay.")

    args = parser.parse_args()

    main()

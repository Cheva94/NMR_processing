#!/usr/bin/python3.6

'''
    Description: corrects phase of CPMG decay and fits it considering 1, 2 or 3
    exponentials and the temporal evolution of parameters. Then plots one decay
    per hour in semilog scale. Also one fit for quarter of experiment progress.
    All the processed data will be also saved in ouput files (.csv).

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coremonitoring_exp import *

def main():

    CPMGs = args.input
    nFiles = len(CPMGs)
    exp = args.exponential_fit
    t_wait = args.t_wait
    back = args.background

    fileRoot = CPMGs[0].split('_0')[0]
    tEvol, tDecay = t_arrays(fileRoot, t_wait, nFiles)

    if back == None:
        if exp == 'mono':
            out_1(tEvol, tDecay, CPMGs, fileRoot, nFiles)
            plot_decay(fileRoot, tEvol, t_wait)
            plot_param1(fileRoot)
        elif exp == 'bi':
            out_2(tEvol, tDecay, CPMGs, fileRoot, nFiles)
            plot_decay(fileRoot, tEvol, t_wait)
            plot_param2(fileRoot)
        elif exp == 'tri':
            out_3(tEvol, tDecay, CPMGs, fileRoot, nFiles)
            plot_decay(fileRoot, tEvol, t_wait)
            plot_param3(fileRoot)
        else:
            print('Must choose number of components to fit: mono, bi or tri.')
    else:
        print('WIP')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Corrects phase of CPMG decay and fits it considering 1, 2 or 3 exponentials, considering temporal evolution of parameters. Then plots one decay per hour in semilog scale. Also one fit for quarter of experiment progress. All the processed data will be also saved in ouput files (.csv).')

    parser.add_argument('input', help = "Path to the inputs file.", nargs = '+')

    parser.add_argument('t_wait', help = "Waiting time (in minutes) between CPMG experiments.", type=int)

    parser.add_argument('exponential_fit', help = "Fits exponential decay. Must choose mono, bi or tri to fit with 1, 2 or 3 exponentials, respectively.")

    parser.add_argument('-back', '--background', help = "Substracts the file given to the input file. It is NOT assumed that the background is already processed.")

    args = parser.parse_args()

    main()

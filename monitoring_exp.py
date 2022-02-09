#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coremonitoring_exp import *

def main():

    FileArr = args.input
    Out = args.output
    exp = args.exponential_fit
    t_wait = args.t_wait

    nF = len(FileArr)
    tEvol, tDecay = t_arrays(t_wait, FileArr[0], nF)

    if exp == 'mono':
        out_1(tEvol, tDecay, FileArr, Out, nF)
        plot_decay(Out, tEvol, t_wait)
        plot_param1(Out)
    elif exp == 'bi':
        out_2(tEvol, tDecay, FileArr, Out, nF)
        plot_decay(Out, tEvol, t_wait)
        plot_param2(Out)
    elif exp == 'tri':
        out_3(tEvol, tDecay, FileArr, Out, nF)
        plot_decay(Out, tEvol, t_wait)
        plot_param3(Out)
    else:
        print('Must choose number of components to fit: mono, bi or tri.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the CPMG files.", nargs='+')
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('t_wait', help = "Waiting time (in minutes) between CPMG experiments.", type=int)
    parser.add_argument('exponential_fit', help = "Fits exponential decay. Must choose mono, bi or tri to fit with 1, 2 or 3 exponentials, respectively.")

    args = parser.parse_args()

    main()

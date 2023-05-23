#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
from core.mouse_DT2 import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = 'data2D.dat'
    Out = '../DT2'
#    Out = args.output
    # Out = File.split(".txt")[0]
#    Map = args.mapType
    # Map = 'cpmg'
    Back = args.background
    alpha = args.alpha
    Dmin, Dmax = args.DRange[0], args.DRange[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    cropD, cropT2 = args.croppedValues[0], args.croppedValues[1]

    print(f'Alpha = {alpha}')

    S0, D, T2, tau1, tau2, K1, K2, Z, N1, N2 = DT2_file(File, Dmin, Dmax, T2min, T2max, cropD, cropT2)

    print('Processing 2D-Laplace inversion...')
    S_2D = NLI_FISTA_2D(K1, K2, Z, alpha, S0)
    print(f'2D inversion ready!')

    M1, M2 = fitMag_2D(tau1, tau2, D, T2, S_2D)

    print('Plotting...')
    plot(tau1, tau2, Z, D, T2, S_2D, M1, M2, Out, Dmin, Dmax, T2min, T2max, alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

#    parser.add_argument('input', help = "Path to the SR-CPMG file.")
#    parser.add_argument('output', help = "Path for the output files.")
#    parser.add_argument('mapType', help = "fid, cpmg, fidcpmg", choices=['fid', 'cpmg', 'fidcpmg'])
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.1)
    parser.add_argument('-D', '--DRange', help = "Range to consider for D values.", nargs = 2, type = float, default = [-2, 2])
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [-1, 3])
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of D and T2.", nargs = 2, type = int, default=[0, 1])
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

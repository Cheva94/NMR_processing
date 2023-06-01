#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
from core.SRmaps import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = args.input
#    Out = args.output
    Out = File.split(".txt")[0]
#    Map = args.mapType
    Map = 'cpmg'
    Back = args.background
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    cropT1, cropT2 = args.croppedValues[0], args.croppedValues[1]

    print(f'Alpha = {alpha}')

    if Back == None:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, cropT2new, S01D, K1D = SRmap_file(File, T1min, T1max, T2min, T2max, cropT1, cropT2, Map)
        Z = PhCorr(signal, N1, N2)

        Back = "Nein!"
    else:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, cropT2new, S01D, K1D = SRmap_file(File, T1min, T1max, T2min, T2max, cropT1, cropT2, Map)
        Z = PhCorr(signal, N1, N2)

        _, _, _, _, _, _, _, back, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = SRmap_file(File, T1min, T1max, T2min, T2max, cropT1, cropT2, Map)
        back = PhCorr(back, N1, N2)

        Z -= back

        Back = "Ja!"

    Z = Norm(Z, RG, N1, N2, cropT1, cropT2new)

    print('Processing 2D-Laplace inversion...')
    S_2D = NLI_FISTA_2D(K1, K2, Z, alpha, S0)
    print(f'2D inversion ready!')

    M1, M2 = fitMag_2D(tau1, tau2, T1, T2, S_2D)

    print('Processing 1D-Laplace inversion...')
    S_1D = NLI_FISTA_1D(K1D, Z[:, 0], alpha, S01D)
    print(f'1D inversion ready!')

    print(f'Fitting T1 distribution from 1D-Laplace in time domain...')
    M_1D = fitMag_1D(tau1, T1, S_1D)

    print('Fitting SR with exponential...')
    SR1D_T1, SR1D_M0, SR1D_T1sd, SR1D_r2  = SR1D_fit(tau1, Z[:, 0],T1min, T1max)

    print('Plotting...')
    plot(tau1, tau2, Z, T1, T2, S_2D, M1, M2, Out, T1min, T1max, T2min, T2max, alpha, Back, cropT1, cropT2, Map, nS, RDT, RG, att, RD, p90, p180, tE, nE, SR1D_T1, SR1D_T1sd, SR1D_r2, SR1D_M0, S_1D, M_1D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the SR-CPMG file.")
#    parser.add_argument('output', help = "Path for the output files.")
#    parser.add_argument('mapType', help = "fid, cpmg, fidcpmg", choices=['fid', 'cpmg', 'fidcpmg'])
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T1', '--T1Range', help = "Range to consider for T1 values.", nargs = 2, type = float, default = [0, 4])
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [1, 4])
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of T1 and T2.", nargs = 2, type = int, default=[0, 1])
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

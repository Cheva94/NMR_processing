#!/usr/bin/python3.10
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
    Out = args.output
    Map = args.mapType
    Back = args.background
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    niniT1, niniT2 = args.croppedValues[0], args.croppedValues[1]

    if T1min == 999:
        if Map == 'fid':
            T1min, T1max = 1, 5
        # elif Map == 'cpmg':
        #     T1min, T1max =
        # elif Map == 'fidcpmg':
        #     T1min, T1max =

    if T2min == 999:
        if Map == 'fid':
            T2min, T2max = -1, 1
        # elif Map == 'cpmg':
            # T2min, T2max =
        # elif Map == 'fidcpmg':
            # T2min, T2max =

    print(f'Alpha = {alpha}')
    print('Processing...')

    if Back == None:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, niniT2new = SRmap_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2, Map)
        Z = PhCorr(signal, N1, N2)

        Back = "Nein!"
    else:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, niniT2new = SRmap_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2, Map)
        Z = PhCorr(signal, N1, N2)

        _, _, _, _, _, _, _, back, _, _, _, _, _, _, _, _, _, _, _, _ = SRmap_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2, Map)
        back = PhCorr(back, N1, N2)

        Z -= back

        Back = "Ja!"

    Z = Norm(Z, RG, N1, N2, niniT1, niniT2new)

    S = NLI_FISTA(K1, K2, Z, alpha, S0)
    print(f'Inversion ready!')

    M1, M2 = fitMag(tau1, tau2, T1, T2, S)

    print('Plotting...')
    plot(tau1, tau2, Z, T1, T2, S, M1, M2, Out, T1min, T1max, T2min, T2max, alpha, Back, niniT1, niniT2, Map, nS, RDT, RG, att, RD, p90, p180, tE, nE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the SR-CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('mapType', help = "fid, cpmg, fidcpmg", choices=['fid', 'cpmg', 'fidcpmg'])
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T1', '--T1Range', help = "Range to consider for T1 values.", nargs = 2, type = float, default = [999, 999])
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [999, 999])
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of T1 and T2.", nargs = 2, type = int, default=[0, 0])
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

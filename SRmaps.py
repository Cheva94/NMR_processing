#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import argparse
from core.SRmaps import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = args.input
    Out = args.output
    Map = args.mapType
    nH = args.protonMoles
    RGnorm = args.RGnorm
    newS = args.ManualS
    Back = args.background
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    niniT1, niniT2 = args.niniValues[0], args.niniValues[1]
    nLevel = args.ContourLevels

    print(f'Alpha = {alpha}')

    if Back == None:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2 = SRCPMG_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2)
        Z = PhCorr(signal, N1, N2, niniT1, niniT2)
    else:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2 = SRCPMG_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2)
        Z = PhCorr(signal, N1, N2, niniT1, niniT2)

        _, _, _, _, _, _, _, back, _, _ = SRCPMG_file(Back, T1min, T1max, T2min, T2max, niniT1, niniT2)
        back = PhCorr(back, N1, N2, niniT1, niniT2)

        Z -= back

    Z = Norm(Z, RGnorm, nH)

    if newS == 'off':
        S = NLI_FISTA(K1, K2, Z, alpha, S0)
        np.savetxt(f"{Out}-DomRates.csv", S, delimiter='\t')
    else:
        S = newSS(f"{Out}-DomRates.csv")

    M1, M2 = fitMag(tau1, tau2, T1, T2, S)

    if Back != None:
        Back = "Yes"

    plot(tau1, tau2, Z, T1, T2, S, M1, M2, Out, nLevel, T1min, T1max, T2min, T2max, RGnorm, alpha, Back, nH, niniT1, niniT2, Map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the SR-CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('mapType', help = "fid, cpmg, fidcpmg", choices=['fid', 'cpmg', 'fidcpmg'])
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T1', '--T1Range', help = "Range to consider for T1 values.", nargs = 2, type = float, default = [1, 3])
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [-1.5, 2.5])
    parser.add_argument('-nH', '--protonMoles', type = float, default = 1)
    parser.add_argument('-nLevel', '--ContourLevels', help = "Number of levels to use in the contour plot.", type = int, default = 30)
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T1 and T2.", nargs = 2, type = int, default=[0, 0])
    parser.add_argument('-newS', '--ManualS', default = 'off')
    parser.add_argument('-RGnorm', '--RGnorm', help = "Normalize by RG.", type = int, default = 70)
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

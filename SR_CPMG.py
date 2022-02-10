#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import argparse
from core.coreSR_CPMG import *

def main():

    File = args.input
    Out = args.output
    m = args.mass
    RGnorm = args.RGnorm
    show = args.ShowPlot
    Back = args.background
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    niniT1, niniT2 = args.niniValues[0], args.niniValues[1]
    nLevel = args.ContourLevels

    if Back == None:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2 = SRCPMG_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2)
        Z = PhCorr(signal, N1, N2, niniT1, niniT2)
    else:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2 = SRCPMG_file(File, T1min, T1max, T2min, T2max, niniT1, niniT2)
        Z = PhCorr(signal, N1, N2, niniT1, niniT2)

        _, _, _, _, _, _, _, back, _, _ = SRCPMG_file(Back, T1min, T1max, T2min, T2max, niniT1, niniT2)
        back = PhCorr(back, N1, N2, niniT1, niniT2)

        Z -= back

    Z = Norm(Z, RGnorm, m)
    S = NLI_FISTA(K1, K2, Z, alpha, S0)

    # np.savetxt(f"{Out}-DomTemp.csv", Z, delimiter=',')
    np.savetxt(f"{Out}-DomRates.csv", S, delimiter=',')

    plot_Z(tau1, tau2, Z, Out)
    peaks1x, peaks2x = plot_proj(T1, T2, S, Out)
    plot_map(T1, T2, S, nLevel, Out, peaks1x, peaks2x, T1min, T1max, T2min, T2max, RGnorm, alpha, Back, m)

    if show == 'on':
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the SR-CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.01)
    parser.add_argument('-T1', '--T1Range', help = "Range to consider for T1 values.", nargs = 2, type = int, default = [0, 5])
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = int, default = [0, 5])
    parser.add_argument('-m', '--mass', help = "Sample mass.", type = float, default = 1)
    parser.add_argument('-nLevel', '--ContourLevels', help = "Number of levels to use in the contour plot.", type = int, default = 100)
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T1 and T2.", nargs = 2, type = int, default=[0, 0])
    parser.add_argument('-show', '--ShowPlot', help = "Show plots.", default = 'off')
    parser.add_argument('-RGnorm', '--RGnorm', help = "Normalize by RG.", type = int)
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

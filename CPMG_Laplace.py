#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import argparse
from core.coreCPMG_Laplace import *

def main():

    File = args.input
    Out = args.output
    m = args.mass
    RGnorm = args.RGnorm
    show = args.ShowPlot
    Back = args.background
    alpha = args.alpha
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    niniT2 = args.niniValues

    if Back == None:
        S0, T2, tau, K, decay, nS, RG, p90, att, RD, tEcho, nEcho = CPMG_file(File, T2min, T2max, niniT2)
        Z = PhCorr(decay)
    else:
        S0, T2, tau, K, decay, nS, RG, p90, att, RD, tEcho, nEcho = CPMG_file(File, T2min, T2max, niniT2)
        Z = PhCorr(decay)

        _, _, _, _, back, _, _, _, _, _, _, _ = CPMG_file(Back, T2min, T2max, niniT2)
        back = PhCorr(back)

        Z -= back

    Z = Norm(Z, RGnorm, RG, m)
    S = NLI_FISTA(K, Z, alpha, S0)

    if Back != None:
        Back = "Yes"

    with open(f'{Out}-DomTemp.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s], tEcho [ms], nEcho, Back, m [g] \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD}, {tEcho:.1f}, {nEcho}, {Back}, {m} \n\n')

        f.write("t [ms], Decay \n")
        for i in range(len(tau)):
            f.write(f'{tau[i]:.6f}, {Z[i]:.6f} \n')

    with open(f'{Out}-DomRates.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s], tEcho [ms], nEcho, Back, m [g], Alpha \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD}, {tEcho:.1f}, {nEcho}, {Back}, {m}, {alpha} \n\n')

        f.write("T2 [ms], Distribution \n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}, {S[i]:.6f} \n')

    plot_Z(tau, Z, Out, nS, RG, RGnorm, p90, att, RD, tEcho, nEcho, Back, m)
    plot_distrib(T2, S, Out, alpha, nS, RG, RGnorm, p90, att, RD, tEcho, nEcho, Back, m)


    if show == 'on':
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('alpha', help = "Tikhonov regularization parameter.", type = float)
    parser.add_argument('T2Range', help = "Range to consider for T2 values.", nargs = 2, type = int)
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=0)
    parser.add_argument('-m', '--mass', help = "Sample mass in g.", type = float, default = 1)
    parser.add_argument('-show', '--ShowPlot', help = "Show plots. Default: off", default = 'off')
    parser.add_argument('-RGnorm', '--RGnorm', help = "Normalize by RG. Default: on", default = "on")
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

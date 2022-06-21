#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import argparse
from core.CPMG_Laplace import *

def main():

    File = args.input
    Out = args.output
    nH = args.protonMoles
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

    Z = Norm(Z, RGnorm, RG, nH)
    S = NLI_FISTA(K, Z, alpha, S0)
    M = fitMag(tau, T2, S)

    if Back != None:
        Back = "Yes"

    cumT2 = np.cumsum(S)
    cumT2 /= cumT2[-1]

    with open(f'{Out}.csv', 'w') as f:
        f.write("nS, RG [dB], RGnorm, p90 [us], Attenuation [dB], RD [s], tEcho [ms], nEcho (t [ms]), Back, nH [g], nini \n")
        f.write(f'{nS}, {RG}, {RGnorm}, {p90}, {att}, {RD}, {tEcho:.1f}, {nEcho:.0f} ({tau[-1]}), {Back}, {nH}, {niniT2} \n\n')

        f.write("T2 [ms], Distribution, Cumulative \n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}, {S[i]:.6f}, {cumT2[i]:.6f} \n')

        f.write("\n\nt [ms], Decay, Fit \n")
        for i in range(len(tau)):
            f.write(f'{tau[i]:.6f}, {Z[i]:.6f}, {M[i]:.6f} \n')

    plot(tau, Z, M, T2, S, Out, nS, RG, RGnorm, p90, att, RD, alpha, tEcho, nEcho, Back, nH, cumT2, niniT2)

    if show == 'on':
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 1)
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = int, default = [0, 5])
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=0)
    parser.add_argument('-nH', '--protonMoles', type = float, default = 1)
    parser.add_argument('-show', '--ShowPlot', help = "Show plots. Default: off", default = 'off')
    parser.add_argument('-RGnorm', '--RGnorm', help = "Normalize by RG. Default: on", default = "on")
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

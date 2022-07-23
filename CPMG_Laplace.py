#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2021.
'''

import argparse
from core.CPMG_Laplace import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = args.input
    Out = args.output
    Back = args.background
    alpha = args.alpha
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    nini = args.niniValues

    print(f'Alpha = {alpha}')
    print('Processing...')

    if Back == None:
        S0, T2, tau, K, decay, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, nP = CPMG_file(File, T2min, T2max, nini)
        Z = PhCorr(decay)

        Back = "Nein!"

    else:
        S0, T2, tau, K, decay, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, nP = CPMG_file(File, T2min, T2max, nini)
        Z = PhCorr(decay)

        _, _, _, _, back, _, _, _, _, _, _ = CPMG_file(Back, T2min, T2max, nini)
        back = PhCorr(back)

        Z -= back

        Back = "Ja!"

    Z = Norm(Z, RG)
    S = NLI_FISTA(K, Z, alpha, S0)

    print(f'Inversion ready!')

    M = fitMag(tau, T2, S, nP)

    cumT2 = np.cumsum(S)
    cumT2 /= cumT2[-1]

    print('Saving...')

    with open(f'{Out}_DistribT2.csv', 'w') as f:
        f.write("T2 [ms]\tDistribution\tCumulative \n")
        for i in range(len(T2)):
            f.write(f'{T2[i]:.6f}\t{S[i]:.6f}\t{cumT2[i]:.6f} \n')

    with open(f'{Out}_Decay.csv', 'w') as f:
        f.write("t [ms]\tDecay\tFit \n")
        for i in range(nP):
            f.write(f'{tau[i]:.6f}\t{Z[i]:.6f}\t{M[i]:.6f} \n')

    print('Plotting...')

    plot(tau, Z, M, T2, S, Out, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, alpha, Back, cumT2, nini, T2min, T2max)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [-1.5, 2.5])
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=0)
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

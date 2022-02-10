#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import argparse
from core.coreCPMG_Laplace import *

def main():

    Files = args.input
    alpha = args.TikhonovReg
    nBin = args.RelaxationMesh
    T2min, T2max = args.RangeT2[0], args.RangeT2[1]
    niniT2 = args.niniValues
    mH = args.proton_mass

    S0, T2, tau, K, decay, nS, RG, RD, tEcho, nEcho = CPMG_file(File, Out, nBin, T2min, T2max, niniT2)

    Z = phase_correction(decay)

    if mH == None:
        decay = normalize(decay, RG)
    else:
        decay = normalize(decay, RG, mH)

    np.savetxt(f"{Out}-PhCorrZ.csv", Z, delimiter=',')
    plot_Z(tau, Z, Out)

    S = NLI_FISTA(K, Z, alpha, S0)

    np.savetxt(f"{Out}-Spectrum.csv", S, delimiter=',')

    plot_spec(T2, S, Out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser = argparse.ArgumentParser(description="Corrects phase of CPMG decay and normalizes it considering the receiver gain. It may also normalize by mass of 1H when given. Then fits it considering 1, 2 or 3 exponentials. Finally it plots the decay in normal and semilog scales with the fitting. All the processed data will be also saved in ouput files (.csv). It may substract the background when given. \n\n Notes: doesn't normalize the background by it mass yet (only by RG).")
    #
    parser.add_argument('input', help = "Path to the CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('alpha', help = "Tikhonov regularization parameter.", type = float)
    parser.add_argument('T2Range', help = "Range to consider for T2 values.", nargs = 2, type = int)
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=0)
    parser.add_argument('-m', '--mass', help = "Sample mass.", type = float, default = 1)

    args = parser.parse_args()

    main()

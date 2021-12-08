#!/usr/bin/python3.6

'''
    Description:

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

    for File in Files:
        print(f'Running file: {File}')

        fileRoot = File.split('.txt')[0]

        S0, T2, tau, K, decay, nS, RG, RD, tEcho, nEcho = userfile(File, fileRoot, nBin, T2min, T2max, niniT2)

        Z = phase_correction(decay)

        if mH == None:
            decay = normalize(decay, RG)
        else:
            decay = normalize(decay, RG, mH)

        np.savetxt(f"{fileRoot}-PhCorrZ.csv", Z, delimiter=',')
        plot_Z(tau, Z, fileRoot)

        S = NLI_FISTA(K, Z, alpha, S0)

        np.savetxt(f"{fileRoot}-Spectrum.csv", S, delimiter=',')

        plot_spec(T2, S, fileRoot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser = argparse.ArgumentParser(description="Corrects phase of CPMG decay and normalizes it considering the receiver gain. It may also normalize by mass of 1H when given. Then fits it considering 1, 2 or 3 exponentials. Finally it plots the decay in normal and semilog scales with the fitting. All the processed data will be also saved in ouput files (.csv). It may substract the background when given. \n\n Notes: doesn't normalize the background by it mass yet (only by RG).")
    #
    parser.add_argument('input', help = "Path to the inputs file.", nargs = '+')

    parser.add_argument('-alpha', '--TikhonovReg', help = "Tikhonov regularization parameter.", type = float, default = 1)

    parser.add_argument('-mesh', '--RelaxationMesh', help = "Number of bins in relaxation time grids.", type = int, default=100)

    parser.add_argument('-T2', '--RangeT2', help = "Range to consider for T2 values.", nargs = 2, type = int, default=[0, 5])

    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=0)

    parser.add_argument('-mH', '--proton_mass', help = "Mass of protons in the sample.", type = float)

    args = parser.parse_args()

    main()

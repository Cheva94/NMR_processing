#!/usr/bin/python3.6

'''
    Description:

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: December, 2021.
'''

import argparse
from core.coreSR_CPMG import *

def main():

    File = args.input
    alpha = args.TikhonovReg
    Nx, Ny = args.RelaxationMesh[0], args.RelaxationMesh[1]
    T1min, T1max = args.RangeT1[0], args.RangeT1[1]
    T2min, T2max = args.RangeT2[0], args.RangeT2[1]
    niniT1, niniT2 = args.niniValues[0], args.niniValues[1]
    nLevel = args.ContourLevels

    fileRoot = File.split('.txt')[0]

    S0, T1, T2, tau1, tau2, K1, K2, decay, N1, N2 = userfile(File, fileRoot, Nx, Ny, T1min, T1max, T2min, T2max, niniT1, niniT2)

    Z = phase_correction(decay, N1, N2, niniT1, niniT2)

    np.savetxt(f"{fileRoot}-PhCorrZ.csv", Z, delimiter=',')
    plot_Z(tau1, tau2, Z, fileRoot)

    S = NLI_FISTA(K1, K2, Z, alpha, S0)
    np.savetxt(f"{fileRoot}-2D_Spectrum.csv", S, delimiter=',')

    peaks1x, peaks2x = plot_proj(T1, T2, S, fileRoot)
    plot_map(T1, T2, S, nLevel, fileRoot, peaks1x, peaks2x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser = argparse.ArgumentParser(description="Corrects phase of CPMG decay and normalizes it considering the receiver gain. It may also normalize by mass of 1H when given. Then fits it considering 1, 2 or 3 exponentials. Finally it plots the decay in normal and semilog scales with the fitting. All the processed data will be also saved in ouput files (.csv). It may substract the background when given. \n\n Notes: doesn't normalize the background by it mass yet (only by RG).")
    #
    parser.add_argument('input', help = "Path to the inputs file.")

    parser.add_argument('-alpha', '--TikhonovReg', help = "Tikhonov regularization parameter.", type = float, default = 1)

    parser.add_argument('-nLevel', '--ContourLevels', help = "Number of levels to use in the contour plot.", type = int, default = 100)

    parser.add_argument('-mesh', '--RelaxationMesh', help = "Number of bins in relaxation time grids.", nargs = 2, type = int, default=[100, 100])

    parser.add_argument('-T1', '--RangeT1', help = "Range to consider for T1 values.", nargs = 2, type = int, default=[0, 5])

    parser.add_argument('-T2', '--RangeT2', help = "Range to consider for T2 values.", nargs = 2, type = int, default=[0, 5])

    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T1 and T2.", nargs = 2, type = int, default=[0, 0])

    args = parser.parse_args()

    main()

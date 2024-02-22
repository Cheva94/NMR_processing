#!/usr/bin/python3.10

import argparse
import core_IO as IO
import core_Plot as graph
import os
import numpy as np

def main():

    print('Analysing CPMG raw data...')
    fileDir = args.input
    alpha = args.alpha
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    Out = fileDir.split('/')[0]+'_procCPMG/'
    isExist = os.path.exists(Out)
    if not isExist:
        os.makedirs(Out)

    t, SGL, SW, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho = IO.readCPMG(fileDir)

    S0, T2, K = IO.initKernel1D(nEcho, t, T2min, T2max)
    params = (rf'Acquisition: RDT = {RDT:.2f} $\mu$s | Atten = {att} dB | '
              rf'RG = {RG} dB | nS = {nS} | RD = {RD:.2f} s | '
              rf'p90 = {p90} $\mu$s | p180 = {p180} $\mu$s | '
              rf'tE = {tEcho:.1f} ms | nE = {nEcho}')

    print('Analysing CPMG raw data...')
    SGL = IO.PhCorrFID(SGL)
    Z = SGL.real

    print(f'Fitting with exponentials...')
    Pearson = []

    Mag_1, T2_1, r2 = IO.expFit_1(t, Z)
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    Mag_2, T2_2, r2 = IO.expFit_2(t, Z)
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    dataFit = np.hstack((np.vstack((Mag_1, T2_1, Mag_2, T2_2)), 
                         np.array([Pearson]).T))

    print(f'Starting NLI: Alpha = {alpha}.')
    S, iter = IO.NLI_FISTA_1D(K, Z, alpha, S0)
    if iter < 100000:
        print('Inversion ready!')
    else:
        print('Warning!')
        print('Maximum number of iterations reached!')
        print('Try modifying T2Range and/or alpha settings.')

    print(f'Fitting NLI results in time domain...')
    MLaplace = IO.fitLapMag_1D(t, T2, S, nEcho)

    print('Writing acquisition parameters...')
    IO.writeCPMG_acq(nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, Out)

    print('Writing CPMG processed data...')
    IO.writeCPMG(t, Z, MLaplace, T2, S, Out)

    print('Plotting CPMG processed data...')
    graph.CPMG(t, Z, T2, S, MLaplace, Out, alpha, T2min, T2max, params, 
               dataFit, tEcho)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the FID fileDir.")
    parser.add_argument('-alpha', type = float, default = 0.1, 
                        help = "Tikhonov regularization parameter.")
    parser.add_argument('-T2Range', nargs = 2, type = float, default = [-1, 4], 
                        help = "Range to consider for T2 values.")
    args = parser.parse_args()
    main()
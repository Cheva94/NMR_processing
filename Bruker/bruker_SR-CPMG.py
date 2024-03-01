#!/usr/bin/python3.10

import argparse
import core_IO as IO
import core_Plot as graph
import os

def main():

    print('Analysing SR-CPMG raw data...')
    fileDir = args.input
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    Out = fileDir.split('/')[0]+'_procSR-CPMG/'
    isExist = os.path.exists(Out)
    if not isExist:
        os.makedirs(Out)

    tau1, tau2, SGL, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho = IO.readSRCPMG(fileDir)
    lenIndir = len(tau1)
    S0, T1, T2, K1, K2 = IO.initKernel2D(lenIndir, nEcho, tau1, tau2, T1min, T1max, T2min, T2max)
    params = (rf'Acquisition: RDT = {RDT:.2f} $\mu$s | Atten = {att} dB | '
              rf'RG = {RG} dB | nS = {nS} | RD = {RD:.2f} s | '
              rf'p90 = {p90} $\mu$s | p180 = {p180} $\mu$s | '
              rf'tE = {tEcho:.1f} ms | nE = {nEcho}')

    print('Analysing SR-CPMG raw data...')
    Z = IO.PhCorr2D(SGL, lenIndir, nEcho) # ya sale parte real

    print(f'Starting NLI: Alpha = {alpha}.')
    S, iter = IO.NLI_FISTA_2D(K1, K2, Z, alpha, S0)
    if iter < 100000:
        print('Inversion ready!')
    else:
        print('Warning!')
        print('Maximum number of iterations reached!')
        print('Try modifying T1Range/T2Range and/or alpha settings.')

    print(f'Fitting NLI results in time domain...')
    MLap_SR, MLap_CPMG = IO.fitLapMag_2D(tau1, tau2, T1, T2, S)

    print('Writing acquisition parameters...')
    IO.writeSRCPMG_acq(nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, Out)

    print('Writing SR-CPMG processed data...')
    IO.writeSRCPMG(T1, T2, S, Out)

    print('Plotting SR-CPMG processed data...')
    graph.SRCPMG(tau1, tau2, Z, T1, T2, S, MLap_SR, MLap_CPMG, Out, 
                 alpha, T1min, T1max, T2min, T2max, params, tEcho)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help = "Path to the SR-CPMG fileDir.")
    parser.add_argument('-alpha', type = float, default = 0.1, 
                        help = "Tikhonov regularization parameter.")
    parser.add_argument('-T1Range', nargs = 2, type = float, default = [-1, 4], 
                        help = "Range to consider for T2 values.")
    parser.add_argument('-T2Range', nargs = 2, type = float, default = [-1, 4], 
                        help = "Range to consider for T2 values.")
    args = parser.parse_args()
    main()
#!/usr/bin/python3.10

import argparse
import core_IO as IO
import core_Plot as graph


def main():

    path = args.path
    root = path.split(".txt")[0]
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    
    print('Reading SR-CPMG map raw data...')
    tau1, tau2, SGL, nP1, nP2 = IO.read2Dsgl(path, root)
    RDT, att, RG, nS, RD, p90, p180, tEcho, nEcho, nFID = IO.read2Dparams(root)
    tau2 = tau2[nFID+1:] # Discards FID part of signal + First CPMG point
    S0, T1, T2, K1, K2 = IO.initKernel2D(nP1, nP2, tau1, tau2, 
                                         T1min, T1max, T2min, T2max)
    params = (rf'Acquisition: RDT = {RDT} $\mu$s | Atten = {att} dB | '
              rf'RG = {RG} dB | nS = {nS} | RD = {RD:.2f} s | '
              rf'p90 = {p90} $\mu$s | p180 = {p180} $\mu$s | '
              rf'tE = {tEcho:.1f} ms | nE = {nEcho}')

    print('Analysing CPMG raw data...')
    Z = IO.PhCorr2D(SGL, nP1, nP2)
    Z = IO.NormRG2D(Z, RG, nP1, nP2, nFID)

    print(f'Starting NLI: Alpha = {alpha}.')
    S, iter = IO.NLI_FISTA_2D(K1, K2, Z, alpha, S0)
    if iter < 100000:
        print('Inversion ready!')
    else:
        print('Warning!')
        print('Maximum number of iterations reached!')
        print('Try modifying T2Range and/or alpha settings.')

    print(f'Fitting NLI results in time domain...')
    MLap_SR, MLap_CPMG = IO.fitLapMag_2D(tau1, tau2, T1, T2, S)

    print('Writing CPMG processed data...')
    IO.writeSRCPMG(T1, T2, S, root)

    print('Plotting CPMG processed data...')
    graph.CPMG(t, Z, T2, S, MLaplace, root, alpha, T2min, T2max, params, 
               dataFit, tEcho)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default = 'CPMG.txt', 
                        help = "Path to the CPMG signal.")
    parser.add_argument('-alpha', type = float, default = 0.1, 
                        help = "Tikhonov regularization parameter.")
    parser.add_argument('-T1Range', nargs = 2, type = float, default = [-1, 4], 
                        help = "Range to consider for T1 values.")
    parser.add_argument('-T2Range', nargs = 2, type = float, default = [-1, 4], 
                        help = "Range to consider for T2 values.")
    args = parser.parse_args()
    main()
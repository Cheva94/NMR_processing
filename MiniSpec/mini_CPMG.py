#!/usr/bin/python3.10

import argparse
import mini_IO as IO
import mini_Plot as graph
import numpy as np

def main():

    path = args.path
    root = path.split(".txt")[0]
    alpha = args.alpha
    T2min, T2max = args.T2Range[0], args.T2Range[1]

    print('Reading CPMG raw data...')
    t, SGL, nP, _ = IO.read1Dsgl(path)
    nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho = IO.read1Dparams(root)
    S0, T2, K = IO.initKernel1D(nP, t, T2min, T2max)

    print('Analysing CPMG raw data...')
    SGL = IO.PhCorr1D(SGL)
    Z = SGL.real
    Z = IO.NormRG(Z, RG)

    print(f'Fitting with exponentials...')
    Pearson = []

    Mag_1, T2_1, r2 = IO.expFit_1(t, Z)
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    Mag_2, T2_2, r2 = IO.expFit_2(t, Z)
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    dataFit = np.hstack((np.vstack((Mag_1, T2_1, Mag_2, T2_2)), np.array([Pearson]).T))

    print(f'Starting NLI: Alpha = {alpha}.')
    S, iter = IO.NLI_FISTA(K, Z, alpha, S0)
    if iter < 100000:
        print('Inversion ready!')
    else:
        print('Warning!')
        print('Maximum number of iterations reached!')
        print('Try modifying T2Range and/or alpha settings.')

    print(f'Fitting NLI results in time domain...')
    MLaplace = IO.fitLapMag(t, T2, S, nP)

    # chequeado hasta acá

    S = S[2:-2]
    cumT2 = np.cumsum(S)

    print('Plotting...')
    graph.CPMG(t, Z, MLaplace, T2, S, root, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, alpha, cumT2, T2min, T2max, dataFit)

    print('Writing output...')

    with open(f'{root}_Decay.csv', 'w') as f:
        f.write("t [ms]\tDecay\tFit \n")
        for i in range(nP):
            f.write(f'{t[i]:.6f}\t{Z[i]:.6f}\t{MLaplace[i]:.6f} \n')

    with open(f'{root}_DistribT2.csv', 'w') as f:
        f.write("T2 [ms]\tDistribution\tCumulative (not Norm.) \n")
        for i in range(len(T2[2:-2])):
            f.write(f'{T2[i]:.6f}\t{S[i]:.6f}\t{cumT2[i]:.6f} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', help = "Path to the CPMG signal.", type=str, default = 'CPMG.txt')
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [-1, 4])
    args = parser.parse_args()
    main()
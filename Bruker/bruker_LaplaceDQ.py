#!/usr/bin/python3.10

import argparse
import core_IO as IO
import core_Plot as graph
import numpy as np

def main():

    path = args.path
    root = 'Laplace'
    alpha = args.alpha
    DipMin, DipMax = args.DipRange[0], args.DipRange[1]
    limSup = args.limSup

    print('Reading DQ raw data...')
    vd_us, bu = IO.readDQLaplace(path)
    vd_ms = 0.001 * vd_us
    # no sé si es necesaria la normalización
    # bu /= np.max(bu)
    vdFit = vd_ms[:limSup]
    nP = len(vdFit)
    buFit = bu[:limSup]

    S0, Dip, K = IO.initKernelDQ(nP, vdFit, DipMin, DipMax)

    print(f'Starting NLI: Alpha = {alpha}.')
    S, iter = IO.NLI_FISTA_1D(K, buFit, alpha, S0)
    if iter < 100000:
        print('Inversion ready!')
    else:
        print('Warning!')
        print('Maximum number of iterations reached!')
        print('Try modifying DipRange and/or alpha settings.')

    print(f'Fitting NLI results in time domain...')
    MLaplace = IO.fitLapMag_Dip(vdFit, Dip, S, nP)

    print('Writing DQ processed data...')
    IO.writeDQLap(vdFit, buFit, MLaplace, Dip, S, root)

    print('Plotting DQ processed data...')
    graph.DQLap(vd_us, bu, Dip, S, MLaplace, root, alpha, DipMin, DipMax, limSup)

    print(f'Fitting with exponentials...')

    # Mag_1, Dip_1, r2 = IO.expFit_1(vd[:limSup], bu[:limSup])
    # print(f'R2 = {r2:.6f}')
    # print(Mag_1)
    # print(Dip_1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = "Path to the DQ signal.")
    parser.add_argument('-limSup', help = "Puntos a tomar para el ajuste", 
                        type=int, default=18)
    parser.add_argument('-alpha', type = float, default = 0.1, 
                        help = "Tikhonov regularization parameter.")
    parser.add_argument('-DipRange', nargs = 2, type = float, default = [0, 100], 
                        help = "Range to consider for Dip values.")
    args = parser.parse_args()
    main()
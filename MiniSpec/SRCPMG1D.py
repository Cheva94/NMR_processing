#!/usr/bin/python3.10
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
from core.SRCPMG1D import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = args.input
    Out = args.output
    Map = args.mapType
    Back = args.background
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    cropT1, cropT2 = args.croppedValues[0], args.croppedValues[1]

    if T1min == 999:
        if Map == 'fid':
            T1min, T1max = 1, 5
        # elif Map == 'cpmg':
        #     T1min, T1max =
        # elif Map == 'fidcpmg':
        #     T1min, T1max =

    if T2min == 999:
        if Map == 'fid':
            T2min, T2max = -1, 1
        # elif Map == 'cpmg':
            # T2min, T2max =
        # elif Map == 'fidcpmg':
            # T2min, T2max =

    print(f'Alpha = {alpha}')

    if Back == None:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, cropT2new, S01D, K1DSR, K1DCPMG = SRmap_file(File, T1min, T1max, T2min, T2max, cropT1, cropT2, Map)
        Z = PhCorr(signal, N1, N2)

        Back = "Nein!"
    else:
        S0, T1, T2, tau1, tau2, K1, K2, signal, N1, N2, nS, RDT, RG, att, RD, p90, p180, tE, nE, cropT2new, S01D, K1DSR = SRmap_file(File, T1min, T1max, T2min, T2max, cropT1, cropT2, Map)
        Z = PhCorr(signal, N1, N2)

        _, _, _, _, _, _, _, back, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = SRmap_file(File, T1min, T1max, T2min, T2max, cropT1, cropT2, Map)
        back = PhCorr(back, N1, N2)

        Z -= back

        Back = "Ja!"

    Z = Norm(Z, RG, N1, N2, cropT1, cropT2new)
    offset = np.min(Z[:, 0])
    Z -= offset
    SR = Z[:, 0]
    CPMG = Z[-1, :]
    
    print('Processing 1D-Laplace inversion...')
    SR_Lap = NLI_FISTA_1D(K1DSR, SR, alpha, S01D)
    CPMG_Lap = NLI_FISTA_1D(K1DCPMG, CPMG, alpha, S01D)
    print(f'1D inversion ready!')

    print(f'Fitting T1 distribution from 1D-Laplace in time domain...')
    Mfit_SR = fitMag_SR(tau1, T1, SR_Lap)
    Mfit_CPMG = fitMag_CPMG(tau2, T2, CPMG_Lap)

    print('Fitting SR with exponential...')
    SR1D_T1, SR1D_M0, SR1D_T1sd, SR1D_M0sd, SR1D_r2  = SR1D_fit1cte(tau1, SR,T1min, T1max)
    SR1D_T1bis, SR1D_M0bis, SR1D_y0, SR1D_T1sdbis, SR1D_M0sdbis, SR1D_y0sd, SR1D_r2bis  = SR1D_fit2cte(tau1, SR,T1min, T1max)
    
    print('Fitting CPMG with exponential...')
    CPMG1D_T1, CPMG1D_M0, CPMG1D_T1sd, CPMG1D_M0sd, CPMG1D_r2  = CPMG1D_fit1cte(tau2, CPMG, T2min, T2max)
    CPMG1D_T1bis, CPMG1D_M0bis, CPMG1D_y0, CPMG1D_T1sdbis, CPMG1D_M0sdbis, CPMG1D_y0sd, CPMG1D_r2bis  = CPMG1D_fit2cte(tau2, CPMG, T2min, T2max)

    print('Plotting...')
    plot(tau1, tau2, SR, CPMG, T1, T2, SR_Lap, CPMG_Lap, Mfit_SR, Mfit_CPMG, SR1D_T1, SR1D_M0, SR1D_T1sd, SR1D_M0sd, SR1D_r2, SR1D_T1bis, SR1D_M0bis, SR1D_y0, SR1D_T1sdbis, SR1D_M0sdbis, SR1D_y0sd, SR1D_r2bis, T1min, T1max, Out, CPMG1D_T1, CPMG1D_M0, CPMG1D_T1sd, CPMG1D_M0sd, CPMG1D_r2, CPMG1D_T1bis, CPMG1D_M0bis, CPMG1D_y0, CPMG1D_T1sdbis, CPMG1D_M0sdbis, CPMG1D_y0sd, CPMG1D_r2bis, T2min, T2max)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the SR-CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('mapType', help = "fid, cpmg, fidcpmg", choices=['fid', 'cpmg', 'fidcpmg'])
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T1', '--T1Range', help = "Range to consider for T1 values.", nargs = 2, type = float, default = [999, 999])
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [999, 999])
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of T1 and T2.", nargs = 2, type = int, default=[0, 0])
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()
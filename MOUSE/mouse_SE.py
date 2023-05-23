#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
from core.mouse_SE import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = args.input
    # Out = '../'+File.split(".dat")[0]
    Back = args.background
    alpha = args.alpha
    Dmin, Dmax = args.DRange[0], args.DRange[1]
    nini = args.croppedValues

    print('Processing...')
    if Back == None:
        S0, D, K, bVal, Z, nP, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD, seqtau2, seqtau1 = CPMG_file(File, Dmin, Dmax, nini)

        Back = "Nein!"

    else:
        S0, D, K, bVal, Z, nP, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD, seqtau2, seqtau1 = CPMG_file(File, Dmin, Dmax, nini)

        _, _, _, _, back, _, _, _, _, _, _ = CPMG_file(Back, Dmin, Dmax, nini)

        Z -= back

        Back = "Ja!"

    # Out = f'../t2-0,{int(seqtau2*100)}ms'
    Out = f'../t2-{int(seqtau2)}ms'

    print(f'Alpha = {alpha}')
    S = NLI_FISTA(K, Z, alpha, S0)
    print(f'Inversion ready!')

    print(f'Fitting Laplace in time domain...')
    MLaplace = fitMag(bVal, D, S, nP)

    cumD = np.cumsum(S)

    print(f'Fitting exponentials...')
    popt, perr, r2 = fit_1(bVal, Z)
    Mag_1 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', '', '']
    D_1 = [fr'D = ({popt[1]:.2f}$\pm${perr[1]:.2f}) $10^{-9}\ m^2/s$', '', '']
    Pearson = [f'R2 = {r2:.6f}', '']

    popt, perr, r2 = fit_2(bVal, Z)
    Mag_2 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})', '']
    D_2 = [fr'D = ({popt[1]:.2f}$\pm${perr[1]:.2f}) $10^{-9}\ m^2/s$', fr'D = ({popt[3]:.2f}$\pm${perr[3]:.2f}) $10^{-9}\ m^2/s$', '']
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    popt, perr, r2 = fit_3(bVal, Z)
    Mag_3 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})', fr'M0 = ({popt[4]:.2f}$\pm${perr[4]:.2f})']
    D_3 = [fr'D = ({popt[1]:.2f}$\pm${perr[1]:.2f}) $10^{-9}\ m^2/s$', fr'D = ({popt[3]:.2f}$\pm${perr[3]:.2f}) $10^{-9}\ m^2/s$', fr'D = ({popt[5]:.2f}$\pm${perr[5]:.2f}) $10^{-9}\ m^2/s$']
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    dataFit = np.hstack((np.vstack((Mag_1, D_1, Mag_2, D_2, Mag_3, D_3)), np.array([Pearson]).T))

    print('Plotting...')
    plot(bVal, Z, MLaplace, D, S, Out, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD, seqtau2, seqtau1, alpha, Back, cumD, nini, Dmin, Dmax, dataFit)

    print('Writing output...')

    with open(f'{Out}_Decay.csv', 'w') as f:
        f.write("t [ms]\tDecay\tFit \n")
        for i in range(nP):
            f.write(f'{bVal[i]:.6f}\t{Z[i]:.6f}\t{MLaplace[i]:.6f} \n')

    with open(f'{Out}_DistribD.csv', 'w') as f:
        f.write("D [ms]\tDistribution\tCumulative (not Norm.) \n")
        # for i in range(len(D[2:-2])):
        for i in range(len(D)):
            f.write(f'{D[i]:.6f}\t{S[i]:.6f}\t{cumD[i]:.6f} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the CPMG file.")
#    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.1)
    parser.add_argument('-D', '--DRange', help = "Range to consider for D values.", nargs = 2, type = float, default = [-2, 2])
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of D.", type = int, default=0)
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

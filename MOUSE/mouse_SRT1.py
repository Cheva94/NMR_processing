#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
from core.mouse_SRT1 import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = args.input
    Out = '../SRT1'
    Back = args.background
    alpha = args.alpha
    T1min, T1max = args.T1Range[0], args.T1Range[1]
    nini = args.croppedValues

    print('Processing...')
    if Back == None:
        S0, T1, tau, K, Z, nP, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD = SR_file(File, T1min, T1max, nini)

        Back = "Nein!"

    else:
        S0, T1, tau, K, Z, nP, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD = SR_file(File, T1min, T1max, nini)

        _, _, _, _, back, _, _, _, _, _, _ = SR_file(Back, T1min, T1max, nini)
        back = PhCorr(back)

        Z -= back

        Back = "Ja!"

    print(f'Alpha = {alpha}')
    S = NLI_FISTA(K, Z, alpha, S0)
    print(f'Inversion ready!')

    print(f'Fitting Laplace in time domain...')
    MLaplace = fitMag(tau, T1, S, nP)

    #S = S[2:-2]
    cumT1 = np.cumsum(S)

    print(f'Fitting exponentials...')
    popt, perr, r2 = fit_1(tau, Z)
    Mag_1 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', '', '']
    T1_1 = [fr'T1 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', '', '']
    Pearson = [f'R2 = {r2:.6f}', '']

    popt, perr, r2 = fit_2(tau, Z)
    Mag_2 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})', '']
    T1_2 = [fr'T1 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', fr'T1 = ({popt[3]:.2f}$\pm${perr[3]:.2f}) ms', '']
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    popt, perr, r2 = fit_3(tau, Z)
    Mag_3 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})', fr'M0 = ({popt[4]:.2f}$\pm${perr[4]:.2f})']
    T1_3 = [fr'T1 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', fr'T1 = ({popt[3]:.2f}$\pm${perr[3]:.2f}) ms', fr'T1 = ({popt[5]:.2f}$\pm${perr[5]:.2f}) ms']
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    dataFit = np.hstack((np.vstack((Mag_1, T1_1, Mag_2, T1_2, Mag_3, T1_3)), np.array([Pearson]).T))

    print('Plotting...')
    plot(tau, Z, MLaplace, T1, S, Out, shEcho, tEcho, nEcho, topEcho, nS, pulse, RD, alpha, Back, cumT1, nini, T1min, T1max, dataFit)

    print('Writing output...')

    with open(f'{Out}_buildUp.csv', 'w') as f:
        f.write("t [ms]\tbuildUp\tFit \n")
        for i in range(nP):
            f.write(f'{tau[i]:.6f}\t{Z[i]:.6f}\t{MLaplace[i]:.6f} \n')

    with open(f'{Out}_DistribT1.csv', 'w') as f:
        f.write("T1 [ms]\tDistribution\tCumulative (not Norm.) \n")
        for i in range(len(T1[2:-2])):
            f.write(f'{T1[i]:.6f}\t{S[i]:.6f}\t{cumT1[i]:.6f} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the SR file.")
#    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.01)
    parser.add_argument('-T1', '--T1Range', help = "Range to consider for T1 values.", nargs = 2, type = float, default = [-1, 2])
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of T1.", type = int, default=0)
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

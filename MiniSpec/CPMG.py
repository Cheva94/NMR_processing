#!/usr/bin/python3.10
# -*- coding: utf-8 -*-
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: July, 2022.
'''

import argparse
from core.CPMG import *
import warnings
warnings.filterwarnings("ignore")

def main():

    File = args.input
#    Out = args.output
    Out = File.split(".txt")[0]
    Back = args.background
    alpha = args.alpha
    T2min, T2max = args.T2Range[0], args.T2Range[1]
    nini = args.croppedValues

    print('Processing...')
    if Back == None:
        S0, T2, tau, K, decay, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, nP = CPMG_file(File, T2min, T2max, nini)
        Z = PhCorr(decay)

        Back = "Nein!"

    else:
        S0, T2, tau, K, decay, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, nP = CPMG_file(File, T2min, T2max, nini)
        Z = PhCorr(decay)

        _, _, _, _, back, _, _, _, _, _, _ = CPMG_file(Back, T2min, T2max, nini)
        back = PhCorr(back)

        Z -= back

        Back = "Ja!"

    Z = Norm(Z, RG)

    print(f'Alpha = {alpha}')
    S = NLI_FISTA(K, Z, alpha, S0)
    print(f'Inversion ready!')

    print(f'Fitting Laplace in time domain...')
    MLaplace = fitMag(tau, T2, S, nP)

    S = S[2:-2]
    cumT2 = np.cumsum(S)

    print(f'Fitting exponentials...')
    popt, perr, r2 = fit_1(tau, Z)
    Mag_1 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', '', '']
    T2_1 = [fr'T2 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', '', '']
    Pearson = [f'R2 = {r2:.6f}', '']

    popt, perr, r2 = fit_2(tau, Z)
    Mag_2 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})', '']
    T2_2 = [fr'T2 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', fr'T2 = ({popt[3]:.2f}$\pm${perr[3]:.2f}) ms', '']
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    popt, perr, r2 = fit_3(tau, Z)
    Mag_3 = [fr'M0 = ({popt[0]:.2f}$\pm${perr[0]:.2f})', fr'M0 = ({popt[2]:.2f}$\pm${perr[2]:.2f})', fr'M0 = ({popt[4]:.2f}$\pm${perr[4]:.2f})']
    T2_3 = [fr'T2 = ({popt[1]:.2f}$\pm${perr[1]:.2f}) ms', fr'T2 = ({popt[3]:.2f}$\pm${perr[3]:.2f}) ms', fr'T2 = ({popt[5]:.2f}$\pm${perr[5]:.2f}) ms']
    Pearson.append(f'R2 = {r2:.6f}')
    Pearson.append('')

    dataFit = np.hstack((np.vstack((Mag_1, T2_1, Mag_2, T2_2, Mag_3, T2_3)), np.array([Pearson]).T))

    print('Plotting...')
    plot(tau, Z, MLaplace, T2, S, Out, nS, RDT, RG, att, RD, p90, p180, tEcho, nEcho, alpha, Back, cumT2, nini, T2min, T2max, dataFit)

    print('Writing output...')

    with open(f'{Out}_Decay.csv', 'w') as f:
        f.write("t [ms]\tDecay\tFit \n")
        for i in range(nP):
            f.write(f'{tau[i]:.6f}\t{Z[i]:.6f}\t{MLaplace[i]:.6f} \n')

    with open(f'{Out}_DistribT2.csv', 'w') as f:
        f.write("T2 [ms]\tDistribution\tCumulative (not Norm.) \n")
        for i in range(len(T2[2:-2])):
            f.write(f'{T2[i]:.6f}\t{S[i]:.6f}\t{cumT2[i]:.6f} \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the CPMG file.")
#    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('-alpha', '--alpha', help = "Tikhonov regularization parameter.", type = float, default = 0.001)
    parser.add_argument('-T2', '--T2Range', help = "Range to consider for T2 values.", nargs = 2, type = float, default = [0, 4])
    parser.add_argument('-crop', '--croppedValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=1)
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")

    args = parser.parse_args()

    main()

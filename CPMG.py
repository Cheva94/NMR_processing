#!/usr/bin/python3.6

'''
    Description: corrects phase of CPMG decay and fits it considering 1, 2 or 3 exponentials. Then plots the decay in semilog scale with the fitting. All the processed data will be also saved in ouput files (.csv).
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreCPMG import *

def main():

    F = args.input

    if args.monoexponential:
        t, decay, tEcho = userfile(F)
        decay = phase_correction(decay)
        popt, r2, chi2, M0, T2, M0_SD, T2_SD = fit_1(t, decay)
        fileRoot = F.split(".txt")[0]
        plot_1(t, decay, popt, tEcho, fileRoot)
        out_1(t, decay, tEcho, fileRoot, r2, chi2, M0, T2, M0_SD, T2_SD)
    elif args.biexponential:
        t, decay, tEcho = userfile(F)
        decay = phase_correction(decay)
        popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = fit_2(t, decay)
        fileRoot = F.split(".txt")[0]
        plot_2(t, decay, popt, tEcho, fileRoot)
        out_2(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD)
    elif args.triexponential:
        t, decay, tEcho = userfile(F)
        decay = phase_correction(decay)
        popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = fit_3(t, decay)
        fileRoot = F.split(".txt")[0]
        plot_3(t, decay, popt, tEcho, fileRoot)
        out_3(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD)
    else:
        print('Must choose an option: -exp1, -exp2 or -exp3. Use -h for guidance.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Corrects phase of CPMG decay and fits it considering 1, 2 or 3 exponentials. Then plots the decay in semilog scale with the fitting. All the processed data will be also saved in ouput files (.csv).')

    parser.add_argument('input', help = "Path to the inputs file.")

    parser.add_argument('-exp1', '--monoexponential', action = 'store_true', help = "Fits monoexponential decay.")

    parser.add_argument('-exp2', '--biexponential', action = 'store_true', help = "Fits biexponential decay.")

    parser.add_argument('-exp3', '--triexponential', action = 'store_true', help = "Fits triexponential decay.")

    args = parser.parse_args()

    main()

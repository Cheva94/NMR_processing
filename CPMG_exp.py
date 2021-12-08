#!/usr/bin/python3.6

'''
    Description: corrects phase of CPMG decay and normalizes it considering the
    receiver gain. It may also normalize by mass of 1H when given. Then fits it
    considering 1, 2 or 3 exponentials. Finally it plots the decay in normal and
    semilog scales with the fitting. All the processed data will be also saved
    in ouput files (.csv). It may substract the background when given.

    Notes: doesn't normalize the background by it mass yet (only by RG).

    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreCPMG_exp import *

def main():

    Files = args.input
    exp = args.exponential_fit
    mH = args.proton_mass
    back = args.background

    for CPMG in Files:
        print(f'Running file: {CPMG}')

        t, nP, decay, nS, RG, RD, tEcho, nEcho = userfile(CPMG)
        decay = phase_correction(decay)

        if mH == None:
            decay = normalize(decay, RG)
        else:
            decay = normalize(decay, RG, mH)

        if back == None:
            fileRoot = CPMG.split(".txt")[0]
            if exp == 'mono':
                popt, r2, chi2, M0, T2, M0_SD, T2_SD = fit_1(t, decay)
                plot_1(t, decay, popt, tEcho, fileRoot)
                out_1(t, decay, tEcho, fileRoot, r2, chi2, M0, T2, M0_SD, T2_SD)
            elif exp == 'bi':
                popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = fit_2(t, decay)
                plot_2(t, decay, popt, tEcho, fileRoot)
                out_2(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD)
            elif exp == 'tri':
                popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = fit_3(t, decay)
                plot_3(t, decay, popt, tEcho, fileRoot)
                out_3(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD)
            else:
                print('Must choose number of components to fit: mono, bi or tri.')
        else:
            fileRoot = CPMG.split(".txt")[0]+'-BackSub'

            t_B, nP_B, back, nS_B, RG_B, RD_B, tEcho_B, nEcho_B = userfile(back)
            if nP != nP_B:
                print('Both files must have same number of points. Quitting job.')
                exit()

            back = phase_correction(back)
            back = normalize(back, RG_B)

            decay -= back

            if exp == 'mono':
                popt, r2, chi2, M0, T2, M0_SD, T2_SD = fit_1(t, decay)
                plot_1(t, decay, popt, tEcho, fileRoot)
                out_1(t, decay, tEcho, fileRoot, r2, chi2, M0, T2, M0_SD, T2_SD)
            elif exp == 'bi':
                popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = fit_2(t, decay)
                plot_2(t, decay, popt, tEcho, fileRoot)
                out_2(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD)
            elif exp == 'tri':
                popt, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = fit_3(t, decay)
                plot_3(t, decay, popt, tEcho, fileRoot)
                out_3(t, decay, tEcho, fileRoot, r2, chi2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD)
            else:
                print('Must choose number of components to fit: mono, bi or tri.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Corrects phase of CPMG decay and normalizes it considering the receiver gain. It may also normalize by mass of 1H when given. Then fits it considering 1, 2 or 3 exponentials. Finally it plots the decay in normal and semilog scales with the fitting. All the processed data will be also saved in ouput files (.csv). It may substract the background when given. \n\n Notes: doesn't normalize the background by it mass yet (only by RG).")

    parser.add_argument('input', help = "Path to the inputs file.", nargs='+')

    parser.add_argument('exponential_fit', help = "Fits exponential decay. Must choose mono, bi or tri to fit with 1, 2 or 3 exponentials, respectively.")

    parser.add_argument('-back', '--background', help = "Substracts the file given to the input file. It is NOT assumed that the background is already processed.")

    parser.add_argument('-mH', '--proton_mass', help = "Mass of protons in the sample.", type = float)

    args = parser.parse_args()

    main()

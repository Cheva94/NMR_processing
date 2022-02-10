#!/usr/bin/python3.8
'''
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreCPMG_exp import *

def main():

    File = args.input
    Out = args.output
    exp = args.exponential_fit
    m = args.mass
    RGnorm = args.RGnorm
    show = args.ShowPlot
    Back = args.background
    nini = args.niniValues

    if Back == None:
        t, signal, nS, RG, p90, att, RD, tEcho, nEcho = CPMG_file(File, nini)
        decay = PhCorr(signal)
    else:
        t, signal, nS, RG, p90, att, RD, tEcho, nEcho = CPMG_file(File, nini)
        decay = PhCorr(signal)

        _, back, _, _, _, _, _, _, _, _ = CPMG_file(Back, nini)
        back = PhCorr(back)

        decay -= back

    decay = Norm(decay, RGnorm, RG, m)

    if Back != None:
        Back = "Yes"

    if exp == 'mono':
        popt, r2, M0, T2, M0_SD, T2_SD = fit_1(t, decay)
        out_1(t, decay, tEcho, Out, r2, M0, T2, M0_SD, T2_SD, nS, RG, RGnorm, p90, att, RD, nEcho, Back, m)
        plot_1(t, decay, popt, tEcho, Out, nS, RG, RGnorm, p90, att, RD, nEcho, r2, Back, m, M0, T2)
    elif exp == 'bi':
        popt, r2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD = fit_2(t, decay)
        out_2(t, decay, tEcho, Out, r2, M0_1, T2_1, M0_2, T2_2, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, nS, RG, RGnorm, p90, att, RD, nEcho, Back, m)
        plot_2(t, decay, popt, tEcho, Out, nS, RG, RGnorm, p90, att, RD, nEcho, r2, Back, m, M0_1, T2_1, M0_2, T2_2)
    elif exp == 'tri':
        popt, r2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD = fit_3(t, decay)
        out_3(t, decay, tEcho, Out, r2, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3, M0_1_SD, T2_1_SD, M0_2_SD, T2_2_SD, M0_3_SD, T2_3_SD, nS, RG, RGnorm, p90, att, RD, nEcho, Back, m)
        plot_3(t, decay, popt, tEcho, Out, nS, RG, RGnorm, p90, att, RD, nEcho, r2, Back, m, M0_1, T2_1, M0_2, T2_2, M0_3, T2_3)
    else:
        print('Must choose number of components to fit: mono, bi or tri.')

    if show == 'on':
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the CPMG file.")
    parser.add_argument('output', help = "Path for the output files.")
    parser.add_argument('exponential_fit', help = "Fits exponential decay. Must choose mono, bi or tri to fit with 1, 2 or 3 exponentials, respectively.")
    parser.add_argument('-m', '--mass', help = "Sample mass in g.", type = float, default = 1)
    parser.add_argument('-RGnorm', '--RGnorm', help = "Normalize by RG. Default: on", default = "on")
    parser.add_argument('-show', '--ShowPlot', help = "Show plots. Default: off", default = 'off')
    parser.add_argument('-back', '--background', help = "Path to de FID background file.")
    parser.add_argument('-nini', '--niniValues', help = "Number of values to avoid at the beginning of T2.", type = int, default=0)

    args = parser.parse_args()

    main()

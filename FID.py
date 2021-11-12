#!/usr/bin/python3.6

'''
    Description: corrects phase of FID and normalizes it considering the receiver gain and number of protons in the sample. Then plots the FID and transforms it to get spectrum in Hz and ppm. All the processed data will be also saved in ouput files (.csv).  It may substract the background when given.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreFID import *

def main():

    FID = args.input
    # nH = args.nH
    back = args.background
    fileRoot = FID.split(".txt")[0]

    t, nP, DW, FID, nS, RD, RG = userfile(FID)
    FID = phase_correction(FID)
    FID /= RG

    if back == None:
        plot_FID(t, FID, nS, RD, fileRoot)
        out_FID(t, FID, fileRoot)

        freq, spec, max_peak = spectrum(FID, nP, DW)
        plot_spec(freq, spec, max_peak, nS, RD, fileRoot)
        out_spec(freq, spec, fileRoot)

    else:
        t_B, nP_B, DW_B, back, nS_B, RD_B, RG_B = userfile(back)
        if nP != nP_B:
            exit()

        back = phase_correction(back)
        back /= RG_B

        FID = back_subs(FID, back)

        plot_FID_B(t, FID, nS, RD, fileRoot)
        out_FID_B(t, FID, fileRoot)

        freq, spec, max_peak = spectrum(FID, nP, DW)
        plot_spec_B(freq, spec, max_peak, nS, RD, fileRoot)
        out_spec_B(freq, spec, fileRoot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Corrects phase of FID and normalizes it considering the receiver gain and number of protons in the sample. Then plots it and transforms it to get spectrum in Hz and ppm. All the processed data will be also saved in ouput files (.csv). It may substract the background when given.')

    parser.add_argument('input', help = "Path to the input file.")

    # parser.add_argument('nH', help = "Number of protons in the sample.", type = int)

    parser.add_argument('-back', '--background', help = "Substracts the file given to the input file. It is NOT assumed that the background is already processed.")

    args = parser.parse_args()

    main()

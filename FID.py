#!/usr/bin/python3.6

'''
    Description: corrects phase of FID and normalizes it considering the receiver gain and number of protons in the sample. Then plots the FID and transforms it to get spectrum in Hz and ppm. All the processed data will be also saved in ouput files (.csv).
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreFID import *

def main():

    F = args.input
    nH = args.nH
    back = args.background
    t, nP, DW, FID, nS, RD, RG = userfile(F)

    FID = phase_correction(FID)
    FID = normalize(FID, nH, RG)
    fileRoot = F.split(".txt")[0]

    # if back == None:
    plot_FID(t, FID, nS, RD, fileRoot)
    out_FID(t, FID, fileRoot)

    freq, spec, max_peak = spectrum(FID, nP, DW)
    plot_spec(freq, spec, max_peak, nS, RD, fileRoot)
    out_spec(freq, spec, fileRoot)

    # else:
    #     back = pd.read_csv(F, header = None, delim_whitespace = True).to_numpy()
    #     FID = background(FID, back)
    #
    #     plot_FID(t, FID, nS, RD, fileRoot)
    #     out_FID(t, FID, fileRoot)
    #
    #     freq, spec, max_peak = spectrum(FID, nP, DW)
    #     plot_spec(freq, spec, max_peak, nS, RD, fileRoot)
    #     out_spec(freq, spec, fileRoot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Corrects phase of FID and normalizes it considering the receiver gain and number of protons in the sample. Then plots it and transforms it to get spectrum in Hz and ppm. All the processed data will be also saved in ouput files.') # It may substract the background when given.')

    parser.add_argument('input', help = "Path to the input file.")

    parser.add_argument('nH', help = "Number of protons in the sample.", type = int)

    # parser.add_argument('-back', '--background', help = "Substracts the file given to the input file. It is assumed that the background is already processed.")

    args = parser.parse_args()

    main()

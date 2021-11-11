#!/usr/bin/python3.6

'''
    Description: plots FID and its spectrum (Hz or ppm), with phase correction.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreFID import *

def main():

    F = args.input
    nH = args.nH
    t, nP, DW, FID, nS, RD, RG = userfile(F)

    FID = phase_correction(FID)
    FID = normalize(FID, nH, RG)
    fileRoot = F.split(".txt")[0]
    plot_FID(t, FID, nS, RD, fileRoot)
    out_FID(t, FID, fileRoot)

    freq, spec, max_peak = spectrum(FID, nP, DW)
    plot_spec(freq, spec, max_peak, nS, RD, fileRoot)
    out_spec(freq, spec, fileRoot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help = "Path to the input file.")

    parser.add_argument('nH', help = "Number of protons in the sample.", type = int)

    args = parser.parse_args()

    main()

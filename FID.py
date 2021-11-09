#!/usr/bin/python3.6

'''
    Description: plots FID and its spectrum (Hz or ppm), with phase correction.
    Written by: Ignacio J. Chevallier-Boutell.
    Dated: November, 2021.
'''

import argparse
from core.coreFID import *

def main():

    Files = args.input_file

    for F in Files:

        t, Np, dw, FID, ns, rd, rg = userfile(F)
        FID = phase_correction(FID)
        plot_FID(t, FID, ns, rd, rg, F)

        freq, spec, max_peak = spectrum(FID, Np, dw)
        if args.MiniSpec:
            plot_spec_mini(freq, spec, max_peak, ns, rd, rg, F)
        else:
            plot_spec_freq(freq, spec, max_peak, ns, rd, rg, F)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', help = "Path to the input files.",
                        nargs = '+')

    parser.add_argument('-mini', '--MiniSpec', action = 'store_true', help =
                        "Plots spectrum vs CS [ppm] (Minispec=20MHz), with \
                        phase correction.")

    args = parser.parse_args()

    main()
